#!/usr/bin/env python

# Back-Propagation Neural
# Written In Python.
# Using Pybrain
# Bayron <Bayron.P27@gmail.com>

__author__ = 'bayron'

import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer

class ANN:
    def __init__(self, ni, nh, no, epoch, lr, m):
        self.lr = lr
        self.ni = ni
        self.nh = nh
        self.no = no
        self.epoch = epoch
        self.m = m

    def addSample(self):
        fileN = open('/home/bayron/Downloads/train_temp.csv')
        fileN.readline()
        rawData = fileN.read().split()
        np.random.shuffle(rawData)
        dataSet = {'feature':[], 'target':[]}
        dataSet['feature'] = np.array([item.split(',')[1:-1] for item in rawData]).astype(np.float)
        dataSet['target']  = np.array([item.split(',')[-1:] for item in rawData]).astype(np.float)

        self.minn = dataSet['target'].min()
        self.maxx = dataSet['target'].max()
        #[0 1]
        dataSet['target']  = ((dataSet['target'] - self.minn) / ((self.maxx - self.minn) * 1.0))

        #input features' mean is nearly 0
        #scale ?
        fmean = np.tile(dataSet['feature'].mean(0),(dataSet['feature'].shape[0],1))
        dataSet['feature'] = dataSet['feature'] - fmean

        self.ds = SupervisedDataSet(dataSet['feature'].shape[1],dataSet['target'].shape[1])
        for i in xrange(dataSet['feature'].shape[0]):
            self.ds.addSample(dataSet['feature'][i],dataSet['target'][i])

    def buildAndTrain(self):
        self.net = buildNetwork(self.ni, self.nh, self.no, bias = True, hiddenclass = TanhLayer)
        self.trainer = BackpropTrainer(self.net, self.ds, learningrate = self.lr, momentum = self.m)
        self.trainer.trainEpochs(self.epoch)

    def predict(self):
        testfp = open('/home/bayron/Downloads/test_temp.csv')
        testfp.readline()
        testData = testfp.read().split()
        testSet = np.array([item.split(',')[1:] for item in testData]).astype(np.float)
        fmean = np.tile(testSet.mean(0),(testSet.shape[0],1))
        testSet = testSet - fmean

        #Write to file
        fp = open('/home/bayron/Downloads/res_temp.csv', 'w')
        fp.write('Id,reference')
        for j in xrange(testSet.shape[0]):
            output = self.net.activate(testSet[j])
            fp.write('\n' + str(j) + ',' + str((self.maxx - self.minn) * output + self.minn))
        fp.close()

if __name__ == "__main__":
    a = ANN(384, 10, 1, 10, 0.1, 0.1)
    a.addSample()
    a.buildAndTrain()
    a.predict()