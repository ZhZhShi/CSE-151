from __future__ import division
from numpy import linalg as LA
from heapq import heappush,heappop,nsmallest
import numpy as np
import operator
import random
import math

class Kernel:
    def __init__(self):
        self.trainData = []
        self.testData = []
        self.train1 = [[],[],[],[]]
        self.train2 = [[],[],[],[]]
        self.read()


    def read(self):
        file = open("pa4train.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            if words[1]=="+1":
                self.trainData.append((words[0],1))
            else:
                self.trainData.append((words[0],-1))
        file = open("pa4test.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            if words[1]=="+1":
                self.testData.append((words[0],1))
            else:
                self.testData.append((words[0],-1))

    def training1(self,set):
        for j in range(0,4):
            for i in self.trainData:
                sum = 0
                for feature in set[j]:
                    sum += feature[1] * self.stringKernel1(i[0],feature[0],j+2)
                if sum*i[1]<=0:
                    set[j].append(i)

    def testing1(self):
        for p in range(2,6):
            total = 0
            error = 0
            for i in self.trainData:
                sum = 0
                for feature in self.train1[p-2]:
                    sum +=feature[1] * self.stringKernel1(i[0],feature[0],p)
                if sum*i[1]<=0:
                    error += 1
                total += 1
            print("training error for p = ",p,": ",error/total)
            total = 0
            error = 0
            for i in self.testData:
                sum = 0
                for feature in self.train1[p-2]:
                    sum +=feature[1] * self.stringKernel1(i[0],feature[0],p)
                if sum*i[1]<=0:
                    error += 1
                total += 1
            print("test error for p = ",p,": ",error/total)

    def training2(self,set):
        for j in range(3,4):
            for i in self.trainData:
                sum = 0
                for feature in set[j]:
                    sum += feature[1] * self.stringKernel2(i[0],feature[0],j+2)
                if sum*i[1]<=0:
                    set[j].append(i)

    def testing2(self):
        for p in range(2,6):
            total = 0
            error = 0
            for i in self.trainData:
                sum = 0
                for feature in self.train2[p-2]:
                    sum +=feature[1] * self.stringKernel2(i[0],feature[0],p)
                if sum*i[1]<=0:
                    error += 1
                total += 1
            print("training error for p = ",p,": ",error/total)
            total = 0
            error = 0
            for i in self.testData:
                sum = 0
                for feature in self.train2[p-2]:
                    sum +=feature[1] * self.stringKernel2(i[0],feature[0],p)
                if sum*i[1]<=0:
                    error += 1
                total += 1
            print("test error for p = ",p,": ",error/total)

    def stringKernel1(self,s1,s2,p):
        ret = 0
        for i in range(0, len(s1)-p+1):
            start=0
            while True:
                start = s2.find(s1[i:i+p],start)+1
                if start>0:
                    ret+=1
                else:
                    break
        return ret

    def stringKernel2(self,s1,s2,p):
        ret = 0
        set = []
        for i in range(0,len(s1)-p+1):
            if s1[i:i+p] not in set:
                if s1[i:i+p] in s2:
                    ret+=1
                set.append(s1[i:i+p])
        return ret

    def getHighest(self,set,p):
        dict = {}
        for feature in set:
            for i in range(0,len(feature[0])-p+1):
                if feature[0][i:i+p] not in dict:
                    dict[feature[0][i:i+p]] = feature[1]
                else:
                    if feature[1]>0:
                        dict[feature[0][i:i+p]]+=1
                    else:
                        dict[feature[0][i:i+p]]-=1
        dictp = sorted(dict.items(),key=operator.itemgetter(1))
        for d in dictp:
            print d




if __name__ == '__main__':
    kernel  = Kernel()
    kernel.training1(kernel.train1)
    kernel.testing1()
    kernel.training2(kernel.train2)
    kernel.testing2()
    kernel.getHighest(kernel.train2[3],5)
