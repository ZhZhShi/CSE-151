from __future__ import division
from numpy import linalg as LA
from heapq import heappush,heappop,nsmallest
import numpy as np
import operator
import random
import math

class Boost:

    def __init__(self):
        self.trainingData = []
        self.testData = []
        self.dic = {}
        self.dVal = [[0 for x in range(450)] for y in range(20)]
        self.alpha = []
        self.h = []
        self.chosen = []
        for i in range(4003):
            self.h.append((i,1))
            self.h.append((i,-1))
        self.testRound = [3,4,7,10,15,20]

    def readTraining(self):
        file = open("pa5train.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            feature = []
            for word in words[0:-1]:
                feature.append(int(word))
            self.trainingData.append((feature,int(words[-1])))


    def readTest(self):
        file = open("pa5test.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            feature = []
            for word in words[0:-1]:
                feature.append(int(word))
            self.testData.append((feature,int(words[-1])))

    def readDic(self):
        file = open("pa5dictionary.txt","r")
        lines = file.readlines()
        i = 0
        for line in lines:
            words = line.split()
            self.dic[i] = words[0]
            i+=1

    def train(self):
        current = [1/450 for x in range(450)]
        for t in range(20):
            print "round ",t
            error = 1
            currenth = (-1,-1)
            correct = []
            incorrect = []
            for h in self.h:
                herror = 0
                c1 = []
                c2 = []
                for i in range(450):
                    if (h[1]==1 and self.trainingData[i][0][h[0]]==1 and self.trainingData[i][1]==-1) or \
                            (h[1]==1 and self.trainingData[i][0][h[0]]==0 and self.trainingData[i][1]==1) or \
                            (h[1]==-1 and self.trainingData[i][0][h[0]]==1 and self.trainingData[i][1]==1) or\
                            (h[1]==-1 and self.trainingData[i][0][h[0]]==0 and self.trainingData[i][1]==-1):
                        herror+=current[i]
                        c2.append(i)
                    else:
                        c1.append(i)
                #print herror
                if herror < error:
                    error = herror
                    currenth = h
                    correct = c1
                    incorrect = c2
            #print(error)
            print "weak learner: ",currenth, self.dic[currenth[0]]
            # print len(correct)
            # print len(incorrect)
            self.chosen.append(currenth)
            alpha = 0.5*np.log((1-error)/error)
            self.alpha.append(alpha)
            Z = 0
            for i in correct:
                Z+=current[i]*math.exp(-alpha)
            for i in incorrect:
                Z+=current[i]*math.exp(alpha)
            for i in correct:
                self.dVal[t][i] = current[i]*math.exp(-alpha)/Z
            for i in incorrect:
                self.dVal[t][i] = current[i]*math.exp(alpha)/Z
            current = self.dVal[t]
            print current

    def trainingError(self):
        for t in self.testRound:
            total = 0
            error = 0
            for data in self.trainingData:
                value = 0
                for i in range(t):
                    if self.chosen[i][1]==1 and data[0][self.chosen[i][0]]==1:
                        value += self.alpha[i]
                    if self.chosen[i][1]==1 and data[0][self.chosen[i][0]]==0:
                        value -= self.alpha[i]
                    if self.chosen[i][1]==-1 and data[0][self.chosen[i][0]]==0:
                        value += self.alpha[i]
                    if self.chosen[i][1]==-1 and data[0][self.chosen[i][0]]==1:
                        value -= self.alpha[i]
                if (data[1]==1 and value<0) or (data[1]==-1 and value>0):
                    error+=1
                total+=1
            print "round ",t," training error: ",error/total

    def testError(self):
        for t in self.testRound:
            total = 0
            error = 0
            for data in self.testData:
                value = 0
                for i in range(t):
                    if self.chosen[i][1]==1 and data[0][self.chosen[i][0]]==1:
                        value += self.alpha[i]
                    if self.chosen[i][1]==1 and data[0][self.chosen[i][0]]==0:
                        value -= self.alpha[i]
                    if self.chosen[i][1]==-1 and data[0][self.chosen[i][0]]==0:
                        value += self.alpha[i]
                    if self.chosen[i][1]==-1 and data[0][self.chosen[i][0]]==1:
                        value -= self.alpha[i]
                if (data[1]==1 and value<0) or (data[1]==-1 and value>0):
                    error+=1
                total+=1
            print "round ",t," test error: ",error/total





if __name__ == '__main__':
    boost = Boost()
    boost.readTraining()
    boost.readTest()
    boost.readDic()
    boost.train()
    boost.trainingError()
    boost.testError()
