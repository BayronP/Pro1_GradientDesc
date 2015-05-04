# !/usr/bin/env python
__author__ = 'bayron'

import numpy as np
import scipy as sp

def getDataSet():
    fp = open('/home/bayron/Downloads/train_temp.csv')
    fp.readline()
    dataSet = {'feature':[], 'result': []}
    rawData = fp.read().split()
    feature = np.array([item.split(',')[1:-1] for item in rawData]).astype(np.float)
    result = np.array([item.split(',')[-1:] for item in rawData]).astype(np.float)
    dataSet['feature'] = feature
    dataSet['result'] = result
    #print feature.shape
    trainSet = {'feature':[], 'result': []}
    testSet = {'feature':[], 'result': []}
    trainSet['feature'] = feature[:24480,:]
    trainSet['result'] = result[:24480,:]
    testSet['feature'] = feature[24480:,:]
    testSet['result'] = result[24480:,:]
    return dataSet

def getTestDataSet():
    fp = open('/home/bayron/Downloads/test_temp.csv')
    fp.readline()
    dataSet = {'feature':[]}
    rawData = fp.read().split()
    dataSet['feature'] = np.array([item.split(',')[1:] for item in rawData]).astype(np.float)
    return dataSet

def errorEstimate(ideal, predict):
    temp = (ideal - predict) ** 2
    m = temp.shape[0]
    res = 1/(2.0*m) * (temp.sum())
    return res

def trainData(dataSet, cutoff, lamda):
    # Get the number of the features
    m, n = dataSet['feature'].shape
    tData = np.ones((m, n + 1))
    tData[:,1:] = dataSet['feature']
    # Define alpha, do an init guess
    params = np.zeros((n + 1, 1))
    params[0,0] = 1
    alpha = 0.075 / (m * 1.0)
    step = 0
    totalS = 10000
    cutoff = cutoff * totalS
    #print cutoff
    #sca = 0
    # Training data using gradient descent
    while step <= totalS:
        step = step + 1
        if step == cutoff:
            alpha = alpha * 2 / 3.0
        if step > cutoff and (step - cutoff) % 1000 == 0:
            alpha = alpha * 0.95
        h = tData.dot(params)
        #err = errorEstimate(dataSet['result'], h)
        delta = h - dataSet['result']
        # (n + 1) x m dot m x 1
        temp = alpha * (tData.transpose().dot(delta))
        params = (1 - lamda * alpha) * params - temp
    return params

def writeToFile(hList):
    fp = open('/home/bayron/Downloads/res_temp1.csv', 'w')
    fp.write('Id,reference')
    for i in range(hList.shape[0]):
        fp.write('\n' + str(i) + ',' + str(hList[i,0]))
    fp.close()

def main():
    # Read train data set from file
    trainSet = getDataSet()
    #print trainSet['feature'].shape,trainSet['result'].shape
    #print testSet['feature'].shape, testSet['result'].shape
    cutoff = 0.3
    #for i in range(4):
    p = 100
    # train data
    params = trainData(trainSet, cutoff, p)
    #print params
    # test data and output to test_res.csv
    #testData = testSet['feature']
    #res = testSet['result']
    testData = getTestDataSet()
    m, n = testData['feature'].shape
    tmpData = np.ones((m, n + 1))
    tmpData[:,1:] = testData['feature']
    h = tmpData.dot(params)
    #error = np.sqrt(np.square(h - testData['result']).mean())
    #print error
    writeToFile(h)

if __name__ == '__main__':
    main()