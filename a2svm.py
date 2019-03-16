import math
import cvxopt
import sys
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sys.path.append('libsvm-3.23/python')
from svmutil import *

def readFiles(filename, full = True, enum1 = 0, enum2 = 0):
    rowsX = []
    rowsY = []
    with open(filename) as file:
        reader = csv.reader(file)
        line_count = 0
        for row in reader:
            if((not full) and int(row[len(row)-1]) != enum1 and int(row[len(row)-1]) != enum2):
                continue
            rowsX.append([])
            for j,val in enumerate(row):
                if(j < len(row) - 1):
                    rowsX[line_count].append(float(val)/255)
                else:
                    if(full):
                        rowsY.append(float(val))
                    else:
                        if (float(val) == enum1):
                            rowsY.append(1.0)
                        else :
                            rowsY.append(-1.0)
            line_count += 1
    return np.array(rowsX), np.array(rowsY)

def linearKernel(x1, x2):
    return np.dot(x1,x2)

def gaussianKernel(x, y, gamma = 0.05):
    return np.exp(-1 * gamma * np.linalg.norm(x - y)** 2)

def train(rowsX, rowsY, kernel = linearKernel, C = 1.0):
    '''
        The Equation for cvxopt will be
            min 1/2 * alpha.T * H * alpha - 1.T * alpha
             w
            such that
                - alpha(i) <= 0
                  alpha(i) <= c
                  y.T * alpha = 0

                  GX <= h
                  AX = B
                  i.e G will stack -identity * Alphas | identity * Alphas
                      h will stack 0s and c's
                      A will be y.T and b will be np.zeros mx1 matrix
    '''

    train_size, num_features = rowsX.shape
    kern = np.zeros((train_size, train_size))
    for i in range(train_size):
        for j in range(train_size):
            kern[i,j] = kernel(rowsX[i], rowsX[j])

    P = cvxopt.matrix(np.outer(rowsY,rowsY) * kern)
    Q = cvxopt.matrix(-1 * np.ones(train_size))
    A = cvxopt.matrix(rowsY, (1, train_size))
    B = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.vstack((-1 * np.identity(train_size),np.identity(train_size))))
    H = cvxopt.matrix(np.hstack((np.zeros(train_size), np.ones(train_size) * C)))

    solution = cvxopt.solvers.qp(P,Q,G,H,A,B)
    alphas = np.ravel(solution['x'])
    temp_support_vec = alphas > 1e-8 # Choosing only non zero support vectors
    count = 0
    for i in temp_support_vec:
        if (i == True):
            count += 1
    print('Number of Support Vectors: ', count)

    ind = np.arange(len(alphas))[temp_support_vec]
    alphas = alphas[temp_support_vec]
    alphas_size = len(alphas)
    support_vec = rowsX[temp_support_vec]
    support_vec_y = rowsY[temp_support_vec]

    # Calculating the intercept term
    b = 0.0
    for i in range(alphas_size):
        b = b + (support_vec_y[i] - np.sum(alphas * support_vec_y * kern[ind[i], temp_support_vec]))
    b = b / alphas_size
    print('b: ', b)
    # Calculating w if the kernel is linear
    w = np.zeros(num_features)
    if(kernel == linearKernel):
        for i in range(alphas_size):
            w = w + ( alphas[i] * support_vec[i] * support_vec_y[i])
    return b, w, alphas, support_vec, support_vec_y

def test(rowsX, rowsY, b, w, alphas, support_vec, support_vec_y, kernel = linearKernel):
    '''
        Function to test the trained model
    '''
    tot = len(rowsX)
    y_test = np.zeros(tot)
    if (kernel == linearKernel):
        y_test = np.sign(np.dot(rowsX,w) + b)
        return np.sum(y_test == rowsY)/tot * 100
    else:
        s = 0.0
        for j in range(tot):
            s = 0.0
            for a, sv_y, sv in zip(alphas, support_vec_y, support_vec):
                s = s + a * sv_y * kernel(rowsX[j], sv)
            y_test[j] = s
        return np.sum(np.sign(y_test + b) == rowsY) / tot * 100

def predictUsingLibSvm(rowsX, rowsY, testX, testY, kerneltype = 'LINEAR', C = 1.0):
    prob = svm_problem(rowsY, rowsX)
    tot = len(testX)
    t = 0 if kerneltype == 'LINEAR' else 2
    param = svm_parameter("-s 0 -t " + str(t) + " -c " + str(C))
    model = svm_train(prob, param)
    acc = svm_predict(testY, testX, model)[1][0]
    return acc

def reduceData(fullX, fullY, enum1, enum2):
    '''
        Reduce the full data to filter all the entries except enum1 and enum2.
        Used as Helper Function for Multi Class SVM
    '''
    X = []
    Y = []
    for i in range(len(fullX)):
        if(fullY[i] == enum1 ):
            X.append(fullX[i])
            Y.append(1.0)
        elif(fullY[i] == enum2):
            X.append(fullX[i])
            Y.append(-1.0)
    return np.array(X), np.array(Y)


def testMultiClass(rowx, b, w, alphas, support_vec, support_vec_y, kernel = gaussianKernel):
    '''
        Helper Function in calculating the prediction value using single Example
    '''
    s = 0.0
    for a, sv_y, sv in zip(alphas, support_vec_y, support_vec):
        s = s + a * sv_y * kernel(rowx, sv)
    y_test = s + b
    return  np.sign(y_test)

def findMaxInd(array):
    '''
        Function to find out Index of largest element of array
    '''
    ind = -1
    max_num = -1
    for i,num in enumerate(array):
        if(num >= max_num):
            ind = i
            max_num = num
    return ind

def multiClassSVM(fullX, fullY, testX, testY, class_size = 10 ,kernel = gaussianKernel):
    '''
        Function for multi class SVM using CVXOPT and (k^C2) classes
    '''
    l = []
    acc = 0
    tot = len(testX)
    for i in range(class_size):
        for j in range(i+1, class_size):
            x, y = reduceData(fullX, fullY, i, j)
            print('Training for Class '+str(i) + ' and ' + str(j))
            b, w, a, sv, sv_y = train(x, y, kernel = gaussianKernel)
            l.append((b,w,a,sv,sv_y))

    for k in range(tot):
        count = 0
        test_array = [0 for i in range(class_size)]
        for i in range(class_size):
            for j in range(i+1, class_size):
                sign = testMultiClass(testX[k], l[count][0], l[count][1], l[count][2], l[count][3], l[count][4])
                count += 1
                if (sign == 1):
                    test_array[i] += 1
                else:
                    test_array[j] += 1
        index = findMaxInd(test_array)
        if(index == int(testY[k])):
            acc += 1
    return acc/ tot * 100

def multiCLassSVMLIBSVM(fullX, fullY, testX, testY, kerneltype = 'GAUSSIAN', C = 1.0):
    '''
        Multi Class SVM using LIBSVM
    '''
    prob = svm_problem(fullY, fullX)
    tot = len(fullX)
    t = 2 if kerneltype == 'GAUSSIAN' else 0
    param = svm_parameter('-s 0 -t ' + str(t) + ' -c '+ str(C) + ' -h 0')
    model = svm_train(prob, param)
    label, acc, val = svm_predict(testY, testX, model)
    return acc,label

def draw_confusion_matrix(true_labels, predicted_labels):
    '''
        Function to calculate and draw confusion matrix
    '''
    c_matrix = confusion_matrix(true_labels, predicted_labels)
    print(c_matrix)
    fig = plt.figure()
    plt.imshow(c_matrix)
    plt.colorbar()
    plt.set_cmap('Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')
    plt.show()
    fig.savefig('ConfusionMatrixforMultiSVM.png')

def plot_validation_accuracies(accuracies, c_values):
    '''
        Plotting the accuracies for various C values in validation set
    '''
    fig = plt.figure()
    c_plot = [np.log(i) for i in c_values]
    plt.plot(c_plot,accuracies)
    plt.title('log(C) vs Accuracies for Test set')
    plt.xlabel('log(C) Values')
    plt.ylabel('Accuracies')
    plt.show()
    fig.savefig('ValidationAccVSC_values_Test_set.png')

def validation(fullX, fullY, C_arr, test = False, X = [], Y = []):
    '''
        Function for Validation i.e for selecting best value of C for training
    '''
    tot = int(len(fullX) / 10)
    y = int(len(fullX) * 0.9)
    f_x = fullX[:y]
    f_y = fullY[:y]
    if(not test):
        testY = fullY[-tot:]
        testX = fullX[-tot:]
    else:
        testY = Y
        testX = X
    accs = []
    for c in C_arr:
        acc, _ = multiCLassSVMLIBSVM(f_x, f_y, testX, testY, C = c)
        accs.append(acc[0])
    return accs

if __name__ == '__main__':
    train_file = str(sys.argv[1])
    test_file = str(sys.argv[2])
    binOrMulti = int(sys.argv[3])
    partnum = str(sys.argv[4])
    entry_num = 3
    if(binOrMulti == 0):
        rowsX , rowsY = readFiles(train_file, False, entry_num, (entry_num + 1) % 10)
        testX , testY = readFiles(test_file, False, entry_num, (entry_num + 1) % 10)
        if(partnum == 'a'):
            b, w, a, sv, sv_y = train(rowsX, rowsY)
            acc = test(rowsX, rowsY, b, w, a, sv, sv_y)
            accuracy = test(testX, testY, b, w, a, sv, sv_y)
            np.savetxt('Support_Vectors.txt', sv)
            print('accuracy for training: ', acc)
            print('accuracy for testing: ', accuracy)
        elif(partnum == 'b'):
            b, w, a, sv, sv_y = train(rowsX, rowsY, kernel = gaussianKernel)
            accuracy = test(testX, testY, b, w, a, sv, sv_y, kernel = gaussianKernel)
            print('accuracy: ', accuracy)
        elif(partnum == 'c'):
            accuracy = predictUsingLibSvm(rowsX, rowsY, testX, testY, kerneltype = 'LINEAR')
            accuracz = predictUsingLibSvm(rowsX, rowsY, testX, testY, kerneltype = 'GAUSSIAN')
            print('accuracy for LINEAR kernel: ' + str(accuracy))
            print('accuracy for GAUSSIAN kernel: ' + str(accuracz))
    elif(binOrMulti == 1):
        # fullrowsX , fullrowsY = readFiles(train_file)
        # fulltestX , fulltestY = readFiles(test_file)
        if(partnum == 'a'):
            acc_train = multiClassSVM(fullrowsX, fullrowsY, fullrowsX, fullrowsY)
            accu_test = multiClassSVM(fullrowsX, fullrowsY, fulltestX, fulltestY)
            print('Train accuracy: ', acc_train)
            print('Test accuracy: ', accu_test)
        elif(partnum == 'b'):
            train_acc, lbl = multiCLassSVMLIBSVM(fullrowsX, fullrowsY, fullrowsX, fullrowsY)
            test_accu,lbl1 = multiCLassSVMLIBSVM(fullrowsX, fullrowsY, fulltestX, fulltestY)
            print('Train accuracy: ', train_acc)
            print('Test accuracy: ', test_accu)
        elif(partnum == 'c'):
            acc, lbl = multiCLassSVMLIBSVM(fullrowsX, fullrowsY, fulltestX, fulltestY)
            draw_confusion_matrix(fulltestY, lbl)
        elif(partnum == 'd'):
            c_vals = [1e-5,1e-3,1.0,5.0,10.0]
            acc1 = [9.8, 9.8, 93.18, 94.56, 94.96]
            # accs = validation(fullrowsX, fullrowsY, c_vals)
            # accs1 = validation(fullrowsX, fullrowsY, c_vals, True, fulltestX, fulltestY)
            # plot_validation_accuracies(accs, c_vals)
            plot_validation_accuracies(acc1,c_vals)
            # print('Accuracies for Validation set: ', accs)
            # print('Accuracies for Test set: ', accs1)
    print('DONE!')
