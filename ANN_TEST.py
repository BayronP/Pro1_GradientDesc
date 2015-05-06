#!/usr/bin/env python

# Back-Propagation Neural
# Written In Python.
# Bayron <Bayron.P27@gmail.com>

__author__ = 'bayron'

import numpy as np

class ANN:
    def __init__(self, ni, nh, no, minn, maxx):
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        self.ai = np.ones(self.ni)
        self.ah = np.ones(self.nh)
        self.ao = np.ones(self.no)

        self.wi = np.random.random((self.ni, self.nh)) * 2 - 1
        self.wo = np.random.random((self.nh, self.no)) * 2 - 1

        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))

        self.minn = minn
        self.maxx = maxx

    def scaleVal(self, val):
        return (self.maxx - self.minn) * val + self.minn

    # update ai, ah, ao
    def update(self, inputs):
        self.ai[:-1] = inputs
        temp = (np.tile(self.ai,(self.nh, 1)) * self.wi.transpose()).sum(1)
        self.ah = 1.0 / (1.0 + np.exp(-temp))
        temp = (np.tile(self.ah,(self.no, 1)) * self.wo.transpose()).sum(1)
        self.ao = 1.0 / (1.0 + np.exp(-temp))
        return self.ao

    def backPropagate(self, targets, N, M):
        targets = np.array(targets).flatten()
        output_delta = self.ao * (1 - self.ao) * (targets - self.ao)

        #temp = (np.tile(self.output_dealta,(selfs.nh, 1)) * self.wo).sum(1)
        temp = self.wo.dot(output_delta).flatten()
        hidden_delta = self.ah * (1 - self.ah) * temp

        temp = np.array([self.ah]).transpose().dot(np.array([output_delta]))
        self.wo = self.wo + self.co * M + temp * N
        self.co[:,:] = temp.copy()
        temp = np.array([self.ai]).transpose().dot(np.array([hidden_delta]))
        self.wi = self.wi + self.ci * M + temp * N
        self.ci[:,:] = temp.copy()

        error = 0.5
        error = error + 0.5 * (targets - self.ao) ** 2
        return error

    def trainAnn(self, dataSet, iterators, N, M = 0.1):
        # N for learning rate
        # M for momentum factor
        for i in xrange(iterators):
            error = 0
            for j in xrange(dataSet['feature'].shape[0]):
                self.update(dataSet['feature'][j])
                error = error + self.backPropagate(dataSet['target'][j], N, M)
            if i % 10 == 0:
                print i,error

    def test(self, testSet):
        #fp = open('/home/bayron/Downloads/res_temp.csv', 'w')
        #fp.write('Id,reference')
        self.output = np.ones(testSet['feature'].shape[0])
        for j in xrange(testSet['feature'].shape[0]):
            self.update(testSet['feature'][j])
            self.output[j] = self.scaleVal(self.ao[0])
            #fp.write('\n' + str(j) + ',' + str(self.scaleVal(self.ao[0])))
        #fp.close()
        error = np.sqrt(np.square(self.output - testSet['target']).mean())
        print 'error: ', error

def test():
    rate = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    mont = [0.01,0.05, 0.1,0.2,0.3,0.4]
    for i in range(len(rate)):
        fileN = open('/home/bayron/Downloads/train_temp.csv')
        fileN.readline()
        rawData = fileN.read().split()
        np.random.shuffle(rawData)
        dataSet = {'feature':[], 'target':[]}
        dataSet['feature'] = np.array([item.split(',')[1:-1] for item in rawData]).astype(np.float)
        dataSet['target']  = np.array([item.split(',')[-1:] for item in rawData]).astype(np.float)
        size = dataSet['target'].shape[0]
        trainSet = {'feature':[], 'target':[]}
        trainSet['feature'] = dataSet['feature'][:size - 200]
        trainSet['target']  = dataSet['target'][:size - 200]
        testSet = {'feature':[], 'target':[]}
        testSet['feature'] = dataSet['feature'][-200:]
        testSet['target']  = dataSet['target'][-200:]
        minn = trainSet['target'].min()
        maxx = trainSet['target'].max()
        trainSet['target']  = (trainSet['target'] - minn) / ((maxx - minn) * 1.0)
        nf = trainSet['feature'].shape[1]
        nt = trainSet['target'].shape[1]
        nn = ANN(nf,100,nt, minn, maxx)
        nn.trainAnn(trainSet,60,rate[i])
        #fp = open('/home/bayron/Downloads/test_temp.csv')
        #fp.readline()
        #testSet = {'feature':[]}
        #testData = fp.read().split()
        #testSet['feature'] = np.array([item.split(',')[1:] for item in testData]).astype(np.float)
        nn.test(testSet)

if __name__ == '__main__':
    test()
