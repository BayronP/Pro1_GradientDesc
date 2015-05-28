#!/usr/bin/env python

# Back-Propagation Neural
# Written In Python.
# Bayron <Bayron.P27@gmail.com>

__author__ = 'bayron'

import numpy as np

class ANN:
    def __init__(self, ni, nh, no, minn, maxx):
        self.ni = ni + 1
        self.nh = nh + 1
        self.no = no

        self.ai = np.ones(self.ni)
        self.ah = np.ones(self.nh)
        self.ao = np.ones(self.no)

        #self.wi = np.random.random((self.ni, self.nh)) * 0.01 - 0.005
        #self.wo = np.random.random((self.nh, self.no)) * 0.01 - 0.005
        self.wi = np.random.normal(0,0.05,(self.ni, self.nh))
        self.wo = np.random.normal(0,0.05,(self.nh, self.no))

        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))

        self.minn = minn
        self.maxx = maxx

    def scaleVal(self, val):
        return (self.maxx - self.minn) * ((val + 1.0) / 2.0) + self.minn

    def sigmoid(self, x):
        return np.tanh(x)

    def dsigmoid(self, x):
        return 1.0 - np.square(x)

    # update ai, ah, ao
    def update(self, inputs):
        self.ai[:-1] = inputs.copy()
        temp = (np.tile(self.ai,(self.nh, 1)) * self.wi.transpose()).sum(1)
        #self.ah = np.tanh(temp)
        #print "temp max:", temp.max()
        #print "temp min:", temp.min()
        self.ah = self.sigmoid(temp)
        #self.ah = 1.0 / (np.exp(-temp) + 1.0)
        #print "temp max:", self.ah.max()
        #print "temp min:", self.ah.min()
        #print "temp min:", temp.min()
        temp = (np.tile(self.ah,(self.no, 1)) * self.wo.transpose()).sum(1)
        #print "temp2 max:", temp.max()
        #print "temp2 min:", temp.min()
        #self.ao = np.tanh(temp)
        #self.ao = 1.0 / (np.exp(-temp) + 1.0)
        self.ao = self.sigmoid(temp)
        #print "temp2 max:", temp.max()
        #print "temp2 min:", temp.min()

    def backPropagate(self, targets, N, M):
        targets = np.array(targets).flatten()
        output_delta = self.dsigmoid(self.ao) * (targets - self.ao)

        temp = self.wo.dot(output_delta).flatten()
        hidden_delta = self.dsigmoid(self.ah) * temp

        temp = np.array([self.ah]).transpose().dot(np.array([output_delta]))
        self.wo = self.wo + self.co * M + temp * N
        #print "wo max:", (temp * N).max()
        #print "wo min:", (temp * N).min()
        self.co[:,:] = temp.copy()
        temp = np.array([self.ai]).transpose().dot(np.array([hidden_delta]))
        self.wi = self.wi + self.ci * M + temp * N
        self.ci[:,:] = temp.copy()

    def trainAnn(self, dataSet, iterators = 100, N = 0.1, M = 0.1):
        # N for learning rate
        # M for momentum factor
        for i in xrange(iterators):
            for j in xrange(dataSet['feature'].shape[0]):
                self.update(dataSet['feature'][j])
                self.backPropagate(dataSet['target'][j], N, M)
            #if i >= 150 and N >= 0.01:
            #    N = N * 0.8

    def test(self, testSet):
        fp = open('/home/bayron/Downloads/res_temp_200.csv', 'w')
        fp.write('Id,reference')
        self.output = np.ones(testSet['feature'].shape[0])
        for j in xrange(testSet['feature'].shape[0]):
            self.update(testSet['feature'][j])
            #self.output[j] = self.scaleVal(self.ao[0])
            fp.write('\n' + str(j) + ',' + str(self.scaleVal(self.ao[0])))
        fp.close()
        print self.wi.min(), self.wi.max()
        print self.wo.min(), self.wo.max()

def test():
    fileN = open('/home/bayron/Downloads/train_temp.csv')
    fileN.readline()
    rawData = fileN.read().split()
    np.random.shuffle(rawData)
    dataSet = {'feature':[], 'target':[]}
    dataSet['feature'] = np.array([item.split(',')[1:-1] for item in rawData]).astype(np.float)
    dataSet['target']  = np.array([item.split(',')[-1:] for item in rawData]).astype(np.float)

    minn = dataSet['target'].min()
    maxx = dataSet['target'].max()
    #[-1 1]
    dataSet['target']  = ((dataSet['target'] - minn) / ((maxx - minn) * 1.0)) * 2.0 - 1.0

    #input features' mean is nearly 0
    #scale ?
    fmean = np.tile(dataSet['feature'].mean(0),(dataSet['feature'].shape[0],1))
    #fmaxx = dataSet['feature'].max()
    dataSet['feature'] = dataSet['feature'] - fmean

    nf = dataSet['feature'].shape[1]
    nt = dataSet['target'].shape[1]
    nn = ANN(nf, 60, nt, minn, maxx)
    nn.trainAnn(dataSet)
    fp = open('/home/bayron/Downloads/test_temp.csv')
    fp.readline()
    testSet = {'feature':[]}
    testData = fp.read().split()
    testSet['feature'] = np.array([item.split(',')[1:] for item in testData]).astype(np.float)
    fmean = np.tile(testSet['feature'].mean(0),(testSet['feature'].shape[0],1))
    testSet['feature'] = testSet['feature'] - fmean
    nn.test(testSet)

if __name__ == '__main__':
    test()



#1.use tanh
#2.[-1 1] --> mean: 0 , var: 1
#3.target -->