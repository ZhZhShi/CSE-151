from __future__ import division
from numpy import linalg as LA
from heapq import heappush,heappop,nsmallest
import numpy as np
import operator
import random
import math


class Perceptron:

    def __init__(self):
        self.data = []
        self.data12 = []
        self.testdata = []
        self.trainp = []
        self.trainpE = []
        self.testpE = []
        self.trainva = [[],[],[],[]]
        self.trainvE = []
        self.testvE = []
        self.trainaE = []
        self.testaE = []
        self.q2c = []
        self.dic = []
        self.q3p = []
        self.matrix = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
        self.testcount = [0]*6


    def read(self):
        file = open("pa3train.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            feature = []
            for word in words[0:len(words)-1]:
                feature.append(int(word))
            self.data.append((feature,int(words[-1])))
            if int(words[-1])==1:
                self.data12.append((feature,-1))
            if int(words[-1])==2:
                self.data12.append((feature,1))

    def readtest(self):
        file = open("pa3test.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            feature = []
            for word in words[0:len(words)-1]:
                feature.append(int(word))
            self.testcount[int(words[-1])-1] += 1
            self.testdata.append((feature,int(words[-1])))



    def readDic(self):
        file = open("pa3dictionary.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            self.dic.append(words[0])


    def train1(self):
        w = [0]*819
        for i in range(0,4):
            for data in self.data12:
                if data[1]*np.dot(np.array(w),np.array(data[0]))<=0:
                    w = np.array(w)+data[1]*np.array(data[0])
            self.trainp.append(w)

    def train2(self):
        w = [0]*819
        c = 1
        for i in range(0,4):
            for data in self.data12:
                if data[1]*np.dot(np.array(w),np.array(data[0]))<=0:
                    self.trainva[i].append((w,c))
                    w = np.array(w)+data[1]*np.array(data[0])
                    c = 1
                else:
                    c += 1

    def trainingpError(self):
        total = 0
        error = [0]*4
        for data in self.data12:
            total += 1
            for i in range(0,4):
                #print(data[1]*np.dot(self.trainp[i],np.array(data[0])))
                if data[1]*np.dot(self.trainp[i],np.array(data[0]))<=0:
                    error[i] += 1
        self.trainpE.append(error[0]/total)
        self.trainpE.append(error[1]/total)
        self.trainpE.append(error[2]/total)
        self.trainpE.append(error[3]/total)


    def testpError(self):
        total = 0
        error = [0]*4
        for data in self.testdata:
            if data[1]==1:
                total += 1
                for i in range(0,4):
                    if -1*np.dot(np.array(self.trainp[i]),np.array(data[0]))<=0:
                        error[i] += 1
            if data[1]==2:
                total += 1
                for i in range(0,4):
                    if np.dot(np.array(self.trainp[i]),np.array(data[0]))<=0:
                        error[i] += 1
        self.testpE.append(error[0]/total)
        self.testpE.append(error[1]/total)
        self.testpE.append(error[2]/total)
        self.testpE.append(error[3]/total)


    def trainingvError(self):
        total = 0
        error = [0]*4
        for data in self.data12:
            total += 1
            for i in range(0,4):
                #print(data[1]*np.dot(self.trainp[i],np.array(data[0])))
                t = 0
                for p in self.trainva[i]:
                    t += p[1]*self.sign(np.dot(p[0],np.array(data[0])))
                if self.sign(t)*data[1]<=0:
                    error[i]+=1
        self.trainvE.append(error[0]/total)
        self.trainvE.append(error[1]/total)
        self.trainvE.append(error[2]/total)
        self.trainvE.append(error[3]/total)


    def testvError(self):
        total = 0
        error = [0]*4
        for data in self.testdata:
            if data[1]!=1 and data[1]!=2:
                continue
            total += 1
            for i in range(0,4):
                #print(data[1]*np.dot(self.trainp[i],np.array(data[0])))
                t = 0
                for p in self.trainva[i]:
                    t += p[1]*self.sign(np.dot(p[0],np.array(data[0])))
                if data[1]==1:
                    if self.sign(t)>0:
                        error[i]+=1
                if data[1]==2:
                    if self.sign(t)<0:
                        error[i]+=1
        self.testvE.append(error[0]/total)
        self.testvE.append(error[1]/total)
        self.testvE.append(error[2]/total)
        self.testvE.append(error[3]/total)


    def trainingaError(self):
        total = 0
        error = [0]*4
        for data in self.data12:
            total += 1
            for i in range(0,4):
                #print(data[1]*np.dot(self.trainp[i],np.array(data[0])))
                t = np.array([0]*819)
                for p in self.trainva[i]:
                    t += p[1]*p[0]
                if self.sign(np.dot(t,np.array(data[0])))*data[1]<=0:
                    error[i]+=1
        self.trainaE.append(error[0]/total)
        self.trainaE.append(error[1]/total)
        self.trainaE.append(error[2]/total)
        self.trainaE.append(error[3]/total)


    def testaError(self):
        total = 0
        error = [0]*4
        for data in self.testdata:
            if data[1]!=1 and data[1]!=2:
                continue
            total += 1
            for i in range(0,4):
                #print(data[1]*np.dot(self.trainp[i],np.array(data[0])))
                t = np.array([0]*819)
                for p in self.trainva[i]:
                    t += p[1]*p[0]
                if i==2:
                    self.q2c = t
                if data[1]==1:
                    if self.sign(np.dot(t,np.array(data[0])))>0:
                        error[i]+=1
                if data[1]==2:
                    if self.sign(np.dot(t,np.array(data[0])))<0:
                        error[i]+=1
        self.testaE.append(error[0]/total)
        self.testaE.append(error[1]/total)
        self.testaE.append(error[2]/total)
        self.testaE.append(error[3]/total)


    def q2(self):
        classifier = self.q2c.tolist()
        sort = self.q2c.tolist()
        classifier.sort()
        print("highest three: ")
        print(self.dic[sort.index(classifier[-1])],": ",classifier[-1])
        print(self.dic[sort.index(classifier[-2])],": ",classifier[-2])
        print(self.dic[sort.index(classifier[-3])],": ",classifier[-3])
        print("lowest three: ")
        print(self.dic[sort.index(classifier[0])],": ",classifier[0])
        print(self.dic[sort.index(classifier[1])],": ",classifier[1])
        print(self.dic[sort.index(classifier[2])],": ",classifier[2])


    def q3c(self):
        for i in range(1,7):
            w = np.array([0]*819)
            for data in self.data:
                if data[1]==i:
                    if np.dot(w,np.array(data[0]))<=0:
                        w = w + np.array(data[0])
                else:
                    if -1*np.dot(w,np.array(data[0]))<=0:
                        w = w - np.array(data[0])
            self.q3p.append(w)


    def q3build(self):
        for data in self.testdata:
            ret = []
            for i in range(0,6):
                if np.dot(self.q3p[i],np.array(data[0]))>0:
                    ret.append(i)
            if len(ret)!=1:
                self.matrix[data[1]-1][6] += 1
            else:
                self.matrix[data[1]-1][ret[0]] += 1
        for i in range(0,6):
            print self.matrix[i]
        print(self.testcount)
        for i in range(0,6):
            for j in range(0,7):
                self.matrix[i][j] = self.matrix[i][j]/self.testcount[i]


    def sign(self,x):
        if x>0:
            return 1
        return -1

if __name__ == '__main__':
    perceptron = Perceptron()
    perceptron.read()
    perceptron.readtest()
    perceptron.train1()
    perceptron.trainingpError()
    perceptron.testpError()
    print(perceptron.trainpE)
    print(perceptron.testpE)
    perceptron.train2()
    perceptron.trainingvError()
    perceptron.testvError()
    print(perceptron.trainvE)
    print(perceptron.testvE)
    perceptron.trainingaError()
    perceptron.testaError()
    print(perceptron.trainaE)
    print(perceptron.testaE)
    perceptron.readDic()
    perceptron.q2()
    perceptron.q3c()
    perceptron.q3build()
    for i in range(0,6):
        print perceptron.matrix[i]

