import os, sys
import numpy as np
import argparse
import time
from scipy import stats
from sklearn.svm import LinearSVC
from random import sample
import random
from tqdm import tqdm
from simple_projections import project_onto_chi_square_ball
from create_datasets import data_loader_hiv

import pandas as pd
def compute_gradients_individual(theta, X, y):
    h = predict_prob(theta, X)
    gradient = np.multiply(X, np.transpose([h-y]))
    return gradient

def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((X, intercept), axis=1)

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def loss(h, y):
    return -y * np.log(h) - (1 - y) * np.log(1 - h)

def compute_gradients_vanilla(theta, X, y):  # the vanilla
    h = predict_prob(theta, X)
    gradient = np.dot(X.T, h - y) / y.size
    return gradient

def compute_gradients_focal(theta, X0, X1, y0, y1, gamma):
    h0 = predict_prob(theta, X0)
    h1 = predict_prob(theta, X1)
    w_y_0 = np.power(h0, gamma)
    gradient_y_0 = np.dot(X0.T, np.multiply(w_y_0, h0 - y0 - gamma * (1 - h0) * np.log(1 - h0)))
    w_y_1 = np.power(1 - h1, gamma)
    gradient_y_1 = np.dot(X1.T, np.multiply(w_y_1, h1 - y1 + gamma * h1 * np.log(h1)))
    gradient = (gradient_y_0 + gradient_y_1) / (y0.size + y1.size)
    return gradient

# w is the weight vector (p_i in paper) of each sample
def compute_gradients_dro(theta, X, y, w):
    h = predict_prob(theta, X)
    gradient = np.dot(X.T, np.multiply(w, h - y))
    return gradient

def compute_gradients_tilting(theta, X_1, y_1, X_2, y_2, t):  # TERM
    h_1 = predict_prob(theta, X_1)
    h_2 = predict_prob(theta, X_2)
    gradient1 = np.dot(X_1.T, h_1 - y_1)
    gradient2 = np.dot(X_2.T, h_2 - y_2)
    l_1 = np.mean(loss(h_1, y_1))
    l_2 = np.mean(loss(h_2, y_2))
    l_max = max(l_1, l_2)
    gradient = np.exp((l_1 - l_max) * t) * gradient1 + np.exp((l_2 - l_max) * t) * gradient2
    ZZ = len(y_1) * np.exp(t * (l_1 - l_max)) + len(y_2) * np.exp(t * (l_2 - l_max))
    return gradient / ZZ

def predict_prob(theta, X):
    return sigmoid(np.dot(X, theta))

def predict(theta, X, threshold):
    # advanced ERM
    res = predict_prob(theta, X)
    return (res - threshold + 0.5).round()

def weighting_func(losses, a, b):
    # given losses, return the weights of the losses based on the weighting function with parameters a, b
    # https://papers.nips.cc/paper/9642-on-human-aligned-risk-minimization.pdf
    num_samples = len(losses)
    Fx = np.asarray(sorted(range(len(losses)), key=lambda k: losses[k])) / num_samples
    weights = ((3-3*b) / (a**2-a+1)) * (3 * Fx**2 - 2 * (a+1) * Fx + a) + 1
    return weights

def compute_gradients_hrm(theta, X, y, w):
    h = predict_prob(theta, X)
    gradient = np.dot(X.T, np.multiply(w, h - y)) / y.size
    return gradient
def AUC(posvalue, negvalue):
    auc = 0
    for i in posvalue:
        for j in negvalue:
            if i > j:
                auc+=1
            elif i==j:
                auc+=0.5
    return auc/(len(posvalue)*len(negvalue))

def create_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果不存在，创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功.")
    else:
        print(f"文件夹 '{folder_path}' 已经存在.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',
                        help='run how many times',
                        type=int,
                        default=5)
    parser.add_argument('--obj',
                        help='objective: erm, dro, tilting, fl, hrm, learnreweight',
                        type=str,
                        default='hrm')
    parser.add_argument('--t',
                        help='value of t for tilting',
                        type=float,
                        default=46.272)#53.3
    parser.add_argument('--rho',
                        help='value of rho for minimax (distributionally robust opt work)',
                        type=float,
                        default=2)
    parser.add_argument('--target', type=float, default=0.15)  # erm loss=0.15
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.1)
    parser.add_argument('--num_iters',
                        help='how many iterations of gradient descent',
                        type=int,
                        default=10000)
    parser.add_argument("--validate_freq", type=int, default=250)
    parser.add_argument('--c',
                        help='regularization parameter for linear SVM',
                        type=float,
                        default=2.0)
    parser.add_argument('--gamma',
                        help='parameter for the focal loss',
                        type=float,
                        default=2.0)
    parser.add_argument('--eval_interval',
                        help='eval every how many iterations (of SGD or GD)',
                        type=int,
                        default=1000)
    parser.add_argument('--threshold',
                        help='decision boundary for ERM',
                        type=float,
                        default=0.5)
    parser.add_argument('--distance',
                        help="distribution distance from train dataset to test dataset",
                        type=float,
                        default=0)
    parser.add_argument('--direction', type=str, default = 'up')
    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    obj = parsed['obj']
    rho = parsed['rho']
    t = parsed['t']
    target = parsed['target']
    num_iters = parsed['num_iters']
    lr = parsed['lr']
    c = parsed['c']
    gamma = parsed['gamma']
    validate_fre = parsed['validate_freq']
    interval = parsed['eval_interval']
    threshold = parsed['threshold']
    distance = parsed['distance']
    direction = parsed['direction']
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)
    minlamb = 0.001
    maxlamb = 100

    random_seed = 0
    random.seed(random_seed)

    rholist = [0.25, 0.5, 1, 2, 4]
    targetlist = [0.1, 0.2, 0.3, 0.4, 0.5]
    betalist = [0, 0.2, 0.4, 0.6, 0.8]
    gammalist = [0.25, 0.5, 1, 2, 4]
    clist = [0.5, 1, 2, 4, 8]
    if obj=='dro':
        parameterlist = rholist
    elif obj == 'klrs':
        parameterlist = targetlist
    elif obj == 'fl':
        parameterlist = gammalist
    elif obj == 'hinge':
        parameterlist = clist
    elif obj == 'hrm':
        parameterlist = betalist
    elif obj =='learnreweight':
        parameterlist = [0]

    raresample = pd.read_csv('test_rare_num.csv')

    distancelist = raresample['distance']
    if direction =='up':
        rarenum = raresample['rare_num_up']
    elif direction == 'down':
        rarenum =raresample['rare_nnum_down']

    print(rarenum)

    for parameter in [0.00]:
        all_data, all_labels = data_loader_hiv("data/hiv1/raw", 0)  # have been randomly shuffled using seed i
        positive_index = np.where(all_labels == 1)[0]
        negative_index = np.where(all_labels == 0)[0]
        testpositivepool = sample(list(positive_index), 329)
        testnegativepool = sample(list(negative_index), 330)
        trainval_index = []
        for y in range(len(all_data)):
            if y not in testnegativepool and y not in testpositivepool:
                trainval_index.append(y)
        train_index = sample(trainval_index, int(0.8 * len(all_labels)))

        val_index = list(filter(lambda x: x not in train_index, trainval_index))

        train_X, train_y = all_data[train_index], all_labels[train_index]
        val_X, val_y = all_data[val_index], all_labels[val_index]
        #print(len(val_X))
        if obj=='learnreweight':
            rare_keep_ind = np.where(val_y == 1)[0][:10]
            common_keep_ind = np.where(val_y == 0)[0][:10]
            val_X, val_y = np.concatenate((val_X[common_keep_ind], val_X[rare_keep_ind]), axis=0), \
                np.concatenate((val_y[common_keep_ind], val_y[rare_keep_ind]), axis=0)
        #print(len(val_X))
        theta = np.zeros(len(train_X[0]))
        #print(theta)
        # print("dimension :{}".format(len(theta)))
        mylamb = maxlamb
        start_time = time.time()

        if obj == 'klrs':
            ermfile = pd.read_csv('result/updistance0.00/0.00ermmetric.csv')
            ermtrainloss = ermfile['train_loss']
            target = (1+parameter)*ermtrainloss[0]
        elif obj == 'dro':
            rho = parameter
        elif obj == 'fl':
            gamma = parameter
        elif obj == 'hinge':
            c = parameter
        elif obj == 'hrm':
            beta = parameter

        if obj == 'fl':
            print("focal loss")
            y_0_idx = np.where(train_y == 0)[0]
            y_1_idx = np.where(train_y == 1)[0]
            X0, y0 = train_X[y_0_idx], train_y[y_0_idx]
            X1, y1 = train_X[y_1_idx], train_y[y_1_idx]
        else:
            class1 = np.where(train_y == 1)[0]
            class2 = np.where(train_y == 0)[0]
            X_1, y_1 = train_X[class1], train_y[class1]
            X_2, y_2 = train_X[class2], train_y[class2]

        if obj == 'hinge':
            clf = LinearSVC(C=c, loss='hinge')
            clf.fit(train_X, train_y)
            theta = clf.coef_.flatten()
        else:
            for j in range(num_iters):
                if obj == 'dro':
                    h = predict_prob(theta, train_X)
                    loss_vector = loss(h, train_y)
                    p = project_onto_chi_square_ball(loss_vector, rho)
                    grads_theta = compute_gradients_dro(theta, train_X, train_y, p)
                elif obj == 'tilting':
                    grads_theta = compute_gradients_tilting(theta, X_1, y_1, X_2, y_2, t)
                elif obj == 'fl':
                    grads_theta = compute_gradients_focal(theta, X0, X1, y0, y1, gamma)
                elif obj == 'erm':
                    grads_theta = compute_gradients_vanilla(theta, train_X, train_y)
                elif obj == 'klrs':
                    h_1 = predict_prob(theta, X_1)
                    h_2 = predict_prob(theta, X_2)
                    gradient1 = np.dot(X_1.T, h_1 - y_1)
                    gradient2 = np.dot(X_2.T, h_2 - y_2)
                    l_1 = np.mean(loss(h_1, y_1))
                    l_2 = np.mean(loss(h_2, y_2))
                    l_max = max(l_1, l_2)
                    gradient = np.exp((l_1 - l_max) / mylamb) * gradient1 + np.exp((l_2 - l_max) / mylamb) * gradient2
                    ZZ = len(y_1) * np.exp((l_1 - l_max) / mylamb) + len(y_2) * np.exp((l_2 - l_max) / mylamb)
                    grads_theta = gradient / ZZ
                    inilamb = mylamb
                    if (j + 1) % validate_fre == 0:
                        tiltedloss = mylamb * np.log(
                            len(y_1) / (len(y_1) + len(y_2)) * np.exp(l_1 / mylamb) + len(y_2) / (
                                    len(y_1) + len(y_2)) * np.exp(l_2 / mylamb))
                        # print('tiltedloss:{}, target:{}, mylamb:{}'.format(tiltedloss, target, mylamb))
                        if tiltedloss < target:
                            lowlamb = minlamb
                            highlamb = mylamb
                            while lowlamb / highlamb < 0.99:
                                templamb = (highlamb + lowlamb) / 2
                                temptiletedloss = templamb * np.log(
                                    len(y_1) / (len(y_1) + len(y_2)) * np.exp(l_1 / templamb) + len(y_2) / (
                                            len(y_1) + len(y_2)) * np.exp(l_2 / templamb))
                                if temptiletedloss < target:
                                    mylamb = templamb
                                    highlamb = templamb
                                    if temptiletedloss > target - 0.01:
                                        break
                                else:
                                    lowlamb = templamb

                elif obj=='hrm':
                    h = predict_prob(theta, train_X)
                    loss_vector = loss(h, train_y)
                    p = weighting_func(loss_vector, a=0.5, b=beta)
                    grads_theta = compute_gradients_hrm(theta, train_X, train_y, p)
                elif obj=="learnreweight":
                    grads_train_individual = compute_gradients_individual(theta, train_X, train_y)
                    grads_train = np.mean(grads_train_individual, axis=0)
                    grads_val = compute_gradients_vanilla(theta - lr * grads_train, val_X, val_y)
                    gradients_w = -1 * np.dot(grads_train_individual, grads_val)
                    w = np.maximum(np.ones(len(train_X)) / len(train_X) - 0.001 * gradients_w, 0)
                    w = w / np.sum(w)
                    theta = theta - lr * np.average(grads_train_individual, axis=0, weights=w)
                    #print(theta)
                if obj!='learnreweight':
                    if np.linalg.norm(grads_theta, ord=2) < 1e-60:
                        break
                    if obj == 'klrs':
                        if (j + 1) % validate_fre != 0:

                            if np.abs(inilamb - mylamb) < 0.001:
                                theta = theta - lr * grads_theta
                    else:
                        theta = theta - lr * grads_theta
        end_time = time.time()
        train_time = end_time-start_time
        print('train time for '+obj+str(train_time))
        sys.exit()
        solution_foler = './solution/'
        create_folder(solution_foler)
        if obj == 'erm':
            np.savetxt(solution_foler + 'erm.txt', theta)
        elif obj == 'dro':
            np.savetxt(solution_foler + 'dro{:.2f}.txt'.format(rho), theta)
        elif obj == 'tilting':
            np.savetxt(solution_foler + 'tilting{:.2f}.txt'.format(t), theta)
        elif obj == 'klrs':
            np.savetxt(solution_foler + 'klrs{:.2f}.txt'.format(target), theta)
        elif obj == 'fl':
            np.savetxt(solution_foler + 'fl{:.2f}.txt'.format(gamma), theta)
        elif obj == 'hinge':
            np.savetxt(solution_foler+'hinge{:.2f}.txt'.format(c), theta)
        elif obj == 'hrm':
            np.savetxt(solution_foler+'hrm{:.2f}.txt'.format(beta), theta)
        elif obj == 'learnreweight':
            np.savetxt(solution_foler+'learnreweight.txt', theta)
        if obj == 'hinge':
            preds_train = clf.predict(train_X)
            preds_val = clf.predict(val_X)
            #loss_train = loss(predict_prob(theta, train_X), train_y)
            #loss_val = loss(predict_prob(theta, val_X), val_y)
            class1 = np.where(train_y == 1)
            class2 = np.where(train_y == 0)
            trainX_1, trainy_1 = train_X[class1], train_y[class1]
            trainX_2, trainy_2 = train_X[class2], train_y[class2]
            preds_trainX_1 = clf.predict(trainX_1)
            preds_trainX_2 = clf.predict(trainX_2)
            class1 = np.where(val_y == 1)
            class2 = np.where(val_y == 0)
            valX_1, valy_1 = val_X[class1], val_y[class1]
            valX_2, valy_2 = val_X[class2], val_y[class2]
            preds_valX_1 = clf.predict(valX_1)
            preds_valX_2 = clf.predict(valX_2)
        else:
            preds_train = predict(theta, train_X, threshold)
            preds_val = predict(theta, val_X, threshold)
            loss_train = loss(predict_prob(theta, train_X), train_y)
            loss_val = loss(predict_prob(theta, val_X), val_y)

            class1 = np.where(train_y == 1)
            class2 = np.where(train_y == 0)
            trainX_1, trainy_1 = train_X[class1], train_y[class1]
            trainX_2, trainy_2 = train_X[class2], train_y[class2]

            preds_trainX_1 = predict(theta, trainX_1, threshold)
            preds_trainX_2 = predict(theta, trainX_2, threshold)
            value_trainX_1 = predict_prob(theta, trainX_1)
            value_trainX_2 = predict_prob(theta, trainX_2)
            loss_trainX_1 = loss(value_trainX_1, trainy_1)
            loss_trainX_2 = loss(value_trainX_2, trainy_2)

            class1 = np.where(val_y == 1)
            class2 = np.where(val_y == 0)
            valX_1, valy_1 = val_X[class1], val_y[class1]
            valX_2, valy_2 = val_X[class2], val_y[class2]

            preds_valX_1 = predict(theta, valX_1, threshold)
            preds_valX_2 = predict(theta, valX_2, threshold)
            value_valX_1 = predict_prob(theta, valX_1)
            value_valX_2 = predict_prob(theta, valX_2)
            loss_valX_1 = loss(value_valX_1, valy_1)
            loss_valX_2 = loss(value_valX_2, valy_2)
        print('===============When rare become more================')
        for z in tqdm(range(len(rarenum))):

            mylamblist = []

            train_TPlist = []
            train_FNlist = []
            train_FPlist = []
            train_TNlist = []
            train_precisionlist = []
            train_recalllist = []
            train_TPRlist = []
            train_TNRlist = []
            train_FPRlist = []
            train_F1list = []
            train_MCClist = []
            train_AUClist = []

            val_TPlist = []
            val_FNlist = []
            val_FPlist = []
            val_TNlist = []
            val_precisionlist = []
            val_recalllist = []
            val_TPRlist = []
            val_TNRlist = []
            val_FPRlist = []
            val_F1list = []
            val_MCClist = []
            val_AUClist = []

            train_loss_list = []
            val_loss_list = []

            train_rare_loss_list = []
            val_rare_loss_list = []

            train_common_loss_list = []
            val_common_loss_list = []

            train_accuracies = []
            val_accuracies = []

            train_rare = []

            train_common = []

            test_loss_list = []

            test_rare_loss_list = []

            test_common_loss_list = []

            test_accuracies = []

            test_rare = []

            test_common = []

            test_TPlist = []
            test_FNlist = []
            test_FPlist = []
            test_TNlist = []
            test_precisionlist = []
            test_recalllist = []
            test_TPRlist = []
            test_TNRlist = []
            test_FPRlist = []
            test_F1list = []
            test_MCClist = []
            test_AUClist = []

            begin = time.time()
            result_folder = './result/'+str(direction)+'distance{:.2f}/'.format(distancelist[z])

            create_folder(result_folder)

            for i in range(num_trials):
                random.seed(i)
                testrare_index = sample(list(testpositivepool), rarenum[z])
                testcommon_index = sample(list(testnegativepool), 400-rarenum[z])
                test_index = testrare_index+testcommon_index
                test_X, test_y = all_data[test_index], all_labels[test_index]

                if obj == 'hinge':
                    preds_train = clf.predict(train_X)
                    preds_test = clf.predict(test_X)
                    preds_val = clf.predict(val_X)

                    class1 = np.where(train_y == 1)
                    class2 = np.where(train_y == 0)
                    trainX_1, trainy_1 = train_X[class1], train_y[class1]
                    trainX_2, trainy_2 = train_X[class2], train_y[class2]

                    preds_trainX_1 = clf.predict(trainX_1)
                    preds_trainX_2 = clf.predict(trainX_2)

                    class1 = np.where(val_y == 1)
                    class2 = np.where(val_y == 0)
                    valX_1, valy_1 = val_X[class1], val_y[class1]
                    valX_2, valy_2 = val_X[class2], val_y[class2]

                    preds_valX_1 = clf.predict(valX_1)
                    preds_valX_2 = clf.predict(valX_2)

                    class1 = np.where(test_y == 1)
                    class2 = np.where(test_y == 0)
                    testX_1, testy_1 = test_X[class1], test_y[class1]
                    testX_2, testy_2 = test_X[class2], test_y[class2]

                    preds_testX_1 = clf.predict(testX_1)
                    preds_testX_2 = clf.predict(testX_2)

                    train_TP = (preds_trainX_1 == 1).sum()
                    train_FN = (preds_trainX_1 == 0).sum()
                    train_FP = (preds_trainX_2 == 1).sum()
                    train_TN = (preds_trainX_2 == 0).sum()

                    val_TP = (preds_valX_1 == 1).sum()
                    val_FN = (preds_valX_1 == 0).sum()
                    val_FP = (preds_valX_2 == 1).sum()
                    val_TN = (preds_valX_2 == 0).sum()

                    test_TP = (preds_testX_1 == 1).sum()
                    test_FN = (preds_testX_1 == 0).sum()
                    test_FP = (preds_testX_2 == 1).sum()
                    test_TN = (preds_testX_2 == 0).sum()

                    train_accuracy = (preds_train == train_y).mean()
                    train_accuracies.append(train_accuracy)

                    test_accuracy = (preds_test == test_y).mean()
                    test_accuracies.append(test_accuracy)

                    val_accuracy = (preds_val == val_y).mean()
                    val_accuracies.append(val_accuracy)
                    # 她自己也知道是label=1的是少数类啊，在论文里面写是那个是少数类
                    rare_sample_train = np.where(train_y == 1)[0]
                    rare_sample_test = np.where(test_y == 1)[0]
                    common_sample_train = np.where(train_y == 0)[0]
                    common_sample_test = np.where(test_y == 0)[0]

                    preds_rare = preds_train[rare_sample_train]
                    rare_train_accuracy = (preds_rare == train_y[rare_sample_train]).mean()
                    train_rare.append(rare_train_accuracy)

                    preds_rare = preds_test[rare_sample_test]
                    rare_test_accuracy = (preds_rare == test_y[rare_sample_test]).mean()
                    test_rare.append(rare_test_accuracy)

                    preds_common = preds_train[common_sample_train]
                    common_train_accuracy = (preds_common == train_y[common_sample_train]).mean()
                    train_common.append(common_train_accuracy)

                    preds_common = preds_test[common_sample_test]
                    common_test_accuracy = (preds_common == test_y[common_sample_test]).mean()

                    train_precision = train_TP / (train_TP + train_FP)
                    val_precision = val_TP / (val_TP + val_FP)
                    test_precision = test_TP / (test_TP + test_FP)

                    train_recall = train_TP / (train_TP + train_FN)
                    val_recall = val_TP / (val_TP + val_FN)
                    test_recall = test_TP / (test_TP + test_FN)

                    train_TPR = train_TP / (train_TP + train_FN)
                    val_TPR = val_TP / (val_TP + val_FN)
                    test_TPR = test_TP / (test_TP + test_FN)

                    train_TNR = train_TN / (train_FP + train_TN)
                    val_TNR = val_TN / (val_FP + val_TN)
                    test_TNR = test_TN / (test_FP + test_TN)

                    train_FPR = train_FP / (train_FP + train_TN)
                    val_FPR = val_FP / (val_FP + val_TN)
                    test_FPR = test_FP / (test_FP + test_TN)

                    train_F1 = 2 * train_precision * train_recall / (train_precision + train_recall)
                    val_F1 = 2 * val_precision * val_recall / (val_precision + val_recall)
                    test_F1 = 2 * test_precision * test_recall / (test_precision + test_recall)

                    train_MCC = (train_TP * train_TN - train_FP * train_FN) / (
                                np.sqrt((train_TP + train_FP) * (train_TP + train_FN)) * np.sqrt(
                            (train_TN + train_FP) * (train_TN + train_FN)))

                    val_MCC = (val_TP * val_TN - val_FP * val_FN) / (
                                np.sqrt((val_TP + val_FP) * (val_TP + val_FN)) * np.sqrt(
                            (val_TN + val_FP) * (val_TN + val_FN)))

                    test_MCC = (test_TP * test_TN - test_FP * test_FN) / (
                                np.sqrt((test_TP + test_FP) * (test_TP + test_FN)) * np.sqrt(
                            (test_TN + test_FP) * (test_TN + test_FN)))

                    train_TPlist.append(train_TP)
                    train_FNlist.append(train_FN)
                    train_FPlist.append(train_FP)
                    train_TNlist.append(train_TN)
                    train_precisionlist.append(train_precision)
                    train_recalllist.append(train_recall)
                    train_TPRlist.append(train_TPR)
                    train_TNRlist.append(train_TNR)
                    train_FPRlist.append(train_FPR)
                    train_F1list.append(train_F1)
                    train_MCClist.append(train_MCC)

                    val_TPlist.append(val_TP)
                    val_FNlist.append(val_FN)
                    val_FPlist.append(val_FP)
                    val_TNlist.append(val_TN)
                    val_precisionlist.append(val_precision)
                    val_recalllist.append(val_recall)
                    val_TPRlist.append(val_TPR)
                    val_TNRlist.append(val_TNR)
                    val_FPRlist.append(val_FPR)
                    val_F1list.append(val_F1)
                    val_MCClist.append(val_MCC)

                    test_TPlist.append(test_TP)
                    test_FNlist.append(test_FN)
                    test_FPlist.append(test_FP)
                    test_TNlist.append(test_TN)
                    test_precisionlist.append(test_precision)
                    test_recalllist.append(test_recall)
                    test_TPRlist.append(test_TPR)
                    test_TNRlist.append(test_TNR)
                    test_FPRlist.append(test_FPR)
                    test_F1list.append(test_F1)
                    test_MCClist.append(test_MCC)

                    test_common.append(common_test_accuracy)

                    mylamblist.append(mylamb)

                    columns_list = {'train_rare_acc', 'test_rare_acc', 'train_common_acc', 'test_common_acc',
                                    "train_overall_acc", 'test_overall_acc', \
                                    'train_TP', 'test_TP', 'train_FN', 'test_FN', 'train_FP', 'test_FP', 'train_TN', 'test_TN',
                                    'train_precision', 'test_precision', \
                                    'train_recall', 'test_recall', 'train_TPR', 'test_TPR', 'train_TNR', 'test_TNR',
                                    'train_FPR', 'test_FPR', \
                                    'train_F1', 'test_F1', 'train_MCC', 'test_MCC', }
                    data = {'train_rare_acc': train_rare, 'test_rare_acc': test_rare, 'train_common_acc': train_common,
                            'test_common_acc': test_common, \
                            'train_overall_acc': train_accuracies, 'test_overall_acc': test_accuracies, \
                            'train_TP': train_TPlist, 'val_TP': val_TPlist, \
                            'test_TP': test_TPlist, 'train_FN': train_FNlist, 'val_FN': val_FNlist, 'test_FN': test_FNlist,
                            'train_TN': train_TNlist, 'val_TN': val_TNlist, \
                            'test_TN': test_TNlist, 'train_precision': train_precisionlist, 'val_precision': val_precisionlist,
                            'test_precison': test_precisionlist, \
                            'train_recall': train_recalllist, 'val_recall': val_recalllist, 'test_recall': test_recalllist,
                            'train_TPR': train_TPRlist, \
                            'val_TPR': val_TPRlist, 'test_TPR': test_TPRlist, 'train_F1': train_F1list, 'val_F1': test_F1list,
                            'test_F1': test_F1list, \
                            'train_MCC': train_MCClist, 'val_MCC': val_MCClist, 'test_MCC': test_MCClist,\
                            'mylambda': mylamblist
                            }
                else:
                    preds_train = predict(theta, train_X, threshold)
                    preds_test = predict(theta, test_X, threshold)
                    preds_val = predict(theta, val_X, threshold)

                    loss_train = loss(predict_prob(theta, train_X), train_y)
                    loss_val = loss(predict_prob(theta, val_X), val_y)
                    loss_test = loss(predict_prob(theta, test_X), test_y)
                    #假设我们希望label为1表示的是positive，label为0表示的是negative，应该就是1表示是hiv对象吧

                    class1 = np.where(train_y == 1)
                    class2 = np.where(train_y == 0)
                    trainX_1, trainy_1 = train_X[class1], train_y[class1]
                    trainX_2, trainy_2 = train_X[class2], train_y[class2]

                    preds_trainX_1 = predict(theta, trainX_1, threshold)
                    preds_trainX_2 = predict(theta, trainX_2, threshold)
                    value_trainX_1 = predict_prob(theta, trainX_1)
                    value_trainX_2 = predict_prob(theta, trainX_2)
                    loss_trainX_1 = loss(value_trainX_1, trainy_1)
                    loss_trainX_2 = loss(value_trainX_2, trainy_2)

                    class1 = np.where(val_y==1)
                    class2 = np.where(val_y==0)
                    valX_1, valy_1 = val_X[class1], val_y[class1]
                    valX_2, valy_2 = val_X[class2], val_y[class2]

                    preds_valX_1 = predict(theta, valX_1, threshold)
                    preds_valX_2 = predict(theta, valX_2, threshold)
                    value_valX_1 = predict_prob(theta, valX_1)
                    value_valX_2 = predict_prob(theta, valX_2)
                    loss_valX_1 = loss(value_valX_1, valy_1)
                    loss_valX_2 = loss(value_valX_2, valy_2)

                    class1 = np.where(test_y == 1)
                    class2 = np.where(test_y == 0)
                    #print(class1)
                    #print(class2)
                    testX_1, testy_1 = test_X[class1], test_y[class1]
                    testX_2, testy_2 = test_X[class2], test_y[class2]

                    preds_testX_1 = predict(theta, testX_1, threshold)
                    preds_testX_2 = predict(theta, testX_2, threshold)
                    value_testX_1 = predict_prob(theta, testX_1)
                    value_testX_2 = predict_prob(theta, testX_2)
                    loss_testX_1 = loss(value_testX_1, testy_1)
                    loss_testX_2 = loss(value_testX_2, testy_2)

                    trainexcesslist = []
                    for i in range(len(value_trainX_1)):
                        for j in range(len(value_trainX_2)):
                            trainexcesslist.append(value_trainX_1[i] - value_trainX_2[j])

                    valexcesslist = []
                    for i in range(len(value_valX_1)):
                        for j in range(len(value_valX_2)):
                            valexcesslist.append(value_valX_1[i] - value_valX_2[j])

                    testexcesslist = []
                    for i in range(len(value_testX_1)):
                        for j in range(len(value_testX_2)):
                            testexcesslist.append(value_testX_1[i] - value_testX_2[j])

                    train_TP = (preds_trainX_1 == 1).sum()
                    train_FN = (preds_trainX_1 == 0).sum()
                    train_FP = (preds_trainX_2 == 1).sum()
                    train_TN = (preds_trainX_2 == 0).sum()

                    val_TP = (preds_valX_1 == 1).sum()
                    val_FN = (preds_valX_1 == 0).sum()
                    val_FP = (preds_valX_2 == 1).sum()
                    val_TN = (preds_valX_2 == 0).sum()

                    test_TP = (preds_testX_1 == 1).sum()
                    test_FN = (preds_testX_1 == 0).sum()
                    test_FP = (preds_testX_2 == 1).sum()
                    test_TN = (preds_testX_2 == 0).sum()

                    train_loss_list.append(np.mean(np.array(loss_train)))
                    train_rare_loss_list.append(np.mean(np.array(loss_trainX_1)))
                    train_common_loss_list.append(np.mean(np.array(loss_trainX_2)))

                    val_loss_list.append(np.mean(np.array(loss_val)))
                    val_rare_loss_list.append(np.mean(np.array(loss_valX_1)))
                    val_common_loss_list.append(np.mean(np.array(loss_valX_2)))

                    test_loss_list.append(np.mean(np.array(loss_test)))
                    test_rare_loss_list.append(np.mean(np.array(loss_testX_1)))
                    test_common_loss_list.append(np.mean(np.array(loss_testX_2)))

                    train_accuracy = (preds_train == train_y).mean()
                    train_accuracies.append(train_accuracy)

                    test_accuracy = (preds_test == test_y).mean()
                    test_accuracies.append(test_accuracy)

                    val_accuracy = (preds_val == val_y).mean()
                    val_accuracies.append(val_accuracy)
                    rare_sample_train = np.where(train_y == 1)[0]
                    rare_sample_test = np.where(test_y == 1)[0]
                    common_sample_train = np.where(train_y == 0)[0]
                    common_sample_test = np.where(test_y == 0)[0]

                    preds_rare = preds_train[rare_sample_train]
                    rare_train_accuracy = (preds_rare == train_y[rare_sample_train]).mean()
                    train_rare.append(rare_train_accuracy)

                    preds_rare = preds_test[rare_sample_test]
                    rare_test_accuracy = (preds_rare == test_y[rare_sample_test]).mean()
                    test_rare.append(rare_test_accuracy)

                    preds_common = preds_train[common_sample_train]
                    common_train_accuracy = (preds_common == train_y[common_sample_train]).mean()
                    train_common.append(common_train_accuracy)

                    preds_common = preds_test[common_sample_test]
                    common_test_accuracy = (preds_common == test_y[common_sample_test]).mean()

                    train_precision = train_TP/(train_TP+train_FP)
                    val_precision = val_TP/(val_TP+val_FP)
                    test_precision = test_TP/(test_TP+test_FP)

                    train_recall = train_TP/(train_TP+train_FN)
                    val_recall = val_TP/(val_TP+val_FN)
                    test_recall = test_TP/(test_TP+test_FN)

                    train_TPR = train_TP/(train_TP+train_FN)
                    val_TPR = val_TP / (val_TP + val_FN)
                    test_TPR = test_TP / (test_TP + test_FN)

                    train_TNR = train_TN/(train_FP+train_TN)
                    val_TNR = val_TN / (val_FP + val_TN)
                    test_TNR = test_TN / (test_FP + test_TN)

                    train_FPR = train_FP/(train_FP+train_TN)
                    val_FPR = val_FP / (val_FP + val_TN)
                    test_FPR = test_FP / (test_FP + test_TN)

                    train_F1 = 2*train_precision * train_recall/(train_precision+train_recall)
                    val_F1 = 2*val_precision * val_recall/(val_precision + val_recall)
                    test_F1 = 2*test_precision * test_recall/(test_precision + test_recall)

                    train_MCC = (train_TP*train_TN-train_FP*train_FN)/(np.sqrt((train_TP+train_FP)*(train_TP+train_FN))*np.sqrt((train_TN+train_FP)*(train_TN+train_FN)))

                    val_MCC = (val_TP * val_TN - val_FP * val_FN) /(np.sqrt((val_TP + val_FP) * (val_TP + val_FN)) *np.sqrt((val_TN + val_FP) * (val_TN + val_FN)))

                    test_MCC = (test_TP * test_TN - test_FP * test_FN) /(np.sqrt((test_TP + test_FP) * (test_TP + test_FN)) * np.sqrt((test_TN + test_FP) * (test_TN + test_FN)))

                    train_AUC = AUC(value_trainX_1, value_trainX_2)
                    val_AUC = AUC(value_valX_1, value_valX_2)
                    #print(value_testX_1)
                    #print(value_testX_2)
                    test_AUC = AUC(value_testX_1, value_testX_2)

                    train_TPlist.append(train_TP)
                    train_FNlist.append(train_FN)
                    train_FPlist.append(train_FP)
                    train_TNlist.append(train_TN)
                    train_precisionlist.append(train_precision)
                    train_recalllist.append(train_recall)
                    train_TPRlist.append(train_TPR)
                    train_TNRlist.append(train_TNR)
                    train_FPRlist.append(train_FPR)
                    train_F1list.append(train_F1)
                    train_MCClist.append(train_MCC)
                    train_AUClist.append(train_AUC)

                    val_TPlist.append(val_TP)
                    val_FNlist.append(val_FN)
                    val_FPlist.append(val_FP)
                    val_TNlist.append(val_TN)
                    val_precisionlist.append(val_precision)
                    val_recalllist.append(val_recall)
                    val_TPRlist.append(val_TPR)
                    val_TNRlist.append(val_TNR)
                    val_FPRlist.append(val_FPR)
                    val_F1list.append(val_F1)
                    val_MCClist.append(val_MCC)
                    val_AUClist.append(val_AUC)

                    test_TPlist.append(test_TP)
                    test_FNlist.append(test_FN)
                    test_FPlist.append(test_FP)
                    test_TNlist.append(test_TN)
                    test_precisionlist.append(test_precision)
                    test_recalllist.append(test_recall)
                    test_TPRlist.append(test_TPR)
                    test_TNRlist.append(test_TNR)
                    test_FPRlist.append(test_FPR)
                    test_F1list.append(test_F1)
                    test_MCClist.append(test_MCC)
                    test_AUClist.append(test_AUC)


                    test_common.append(common_test_accuracy)
                    #max_loss.append(l_max)
                    #erm_loss.append(train_loss)
                    mylamblist.append(mylamb)


                    columns_list = {'train_rare_acc', 'test_rare_acc', 'train_common_acc', 'test_common_acc', "train_overall_acc", 'test_overall_acc',\
                        'train_TP', 'test_TP', 'train_FN', 'test_FN', 'train_FP', 'test_FP', 'train_TN', 'test_TN', 'train_precision', 'test_precision',\
                        'train_recall', 'test_recall', 'train_TPR', 'test_TPR', 'train_TNR', 'test_TNR', 'train_FPR', 'test_FPR',\
                        'train_F1', 'test_F1', 'train_MCC', 'test_MCC', }


                    data = {'train_rare_acc': train_rare, 'test_rare_acc': test_rare, 'train_common_acc': train_common, 'test_common_acc': test_common,\
                        'train_overall_acc': train_accuracies, 'test_overall_acc': test_accuracies, \
                        'train_loss': train_loss_list, 'val_loss': val_loss_list, 'test_loss': test_loss_list,\
                        'train_rare_loss': train_rare_loss_list, 'val_rare_loss': val_rare_loss_list, 'test_rare_loss':test_rare_loss_list,\
                        'train_common_loss': train_common_loss_list, 'val_common_loss':val_common_loss_list, 'test_common_loss': test_common_loss_list,\
                        'train_TP':train_TPlist, 'val_TP':val_TPlist, \
                        'test_TP': test_TPlist, 'train_FN': train_FNlist, 'val_FN': val_FNlist, 'test_FN': test_FNlist, 'train_TN': train_TNlist, 'val_TN': val_TNlist,\
                        'test_TN': test_TNlist, 'train_precision': train_precisionlist, 'val_precision': val_precisionlist, 'test_precison': test_precisionlist, \
                        'train_recall': train_recalllist, 'val_recall': val_recalllist, 'test_recall': test_recalllist, 'train_TPR': train_TPRlist,\
                        'val_TPR': val_TPRlist, 'test_TPR':test_TPRlist, 'train_F1': train_F1list, 'val_F1': test_F1list, 'test_F1': test_F1list,\
                        'train_MCC': train_MCClist, 'val_MCC': val_MCClist, 'test_MCC':test_MCClist, 'train_AUC':train_AUClist, "val_AUC":val_AUClist, 'test_AUC':test_AUClist,\
                        'mylambda': mylamblist
                        }

            df = pd.DataFrame(data)

            if obj == 'erm':
                df.to_csv(result_folder+'{:.2f}ermmetric.csv'.format(distancelist[z]), index = False)

            elif obj == 'dro':
                df.to_csv(result_folder+'{:.2f}dro{:.2f}metric.csv'.format(distancelist[z], rho), index = False)

            elif obj == 'tilting':
                df.to_csv(result_folder+'{:.2f}tilting{:.2f}metric.csv'.format(distancelist[z], t), index =False)

            elif obj == 'klrs':
                df.to_csv(result_folder+'{:.2f}klrs{:.2f}metric.csv'.format(distancelist[z], target), index = False)

            elif obj == 'fl':
                df.to_csv(result_folder+'{:.2f}fl{:.2f}metric.csv'.format(distancelist[z], gamma), index = False)

            elif obj == 'hinge':
                df.to_csv(result_folder+'{:.2f}hinge{:.2f}metric.csv'.format(distancelist[z], c), index = False)

            elif obj == 'hrm':
                df.to_csv(result_folder+"{:.2f}hrm{:.2f}metric.csv".format(distancelist[z], beta), index = False)

            elif obj == 'learnreweight':
                df.to_csv(result_folder+"{:.2f}learnreweightmetric.csv".format(distancelist[z]), index = False)

if __name__ == '__main__':
    main()