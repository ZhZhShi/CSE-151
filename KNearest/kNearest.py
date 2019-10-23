from __future__ import division
from numpy import linalg as LA
from heapq import heappush,heappop,nsmallest
import numpy as np
import operator
import random

class Data():
    def __init__(self):
        self.training_data = []
        self.projection_data = []
        self.training_error_1 = {}
        self.validation_error_1 = {}
        self.test_error_1 = {}
        self.training_error_2 = {}
        self.validation_error_2 = {}
        self.test_error_2 = {}
        self.new_training_data = []
        for i in range(0,20):
            self.projection_data.append([])
        #part 1
        print("part1")
        self.read()
        self.checkTraining()
        self.checkValidation()
        self.checkTest()
        #part 2
        print("part2")
        self.calculate(self.training_data,self.new_training_data)
        self.trainingDist()
        self.validationDist()
        self.testDist()
        #print(nsmallest(3,self.training_distance,key=lambda x:abs(x-1500)))

    def read(self):
        file = open("pa1train.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            feature = []
            for word in words[0:len(words)-1]:
                feature.append(int(word))
            self.training_data.append((feature,int(words[-1])))
        file = open("projection.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            for i in range(0,20):
                self.projection_data[i].append(float(words[i]))

    def checkTraining(self):
        file = open("pa1train.txt","r")
        self.check(file,self.training_error_1)
        print(self.training_error_1)

    def checkValidation(self):
        file = open("pa1validate.txt","r")
        self.check(file,self.validation_error_1)
        print(self.validation_error_1)

    def checkTest(self):
        file = open("pa1test.txt","r")
        self.check(file,self.test_error_1)
        print(self.test_error_1)

    def check(self,file,dict):
        ran = [1,3,5,9,15]
        total = {1:0,3:0,5:0,9:0,15:0}
        error = {1:0,3:0,5:0,9:0,15:0}
        lines = file.readlines()
        for line in lines:
            words = line.split()
            feature = []
            for word in words[0:len(words)-1]:
                feature.append(int(word))
            #print(feature)
            distance = []
            for data in self.training_data:
                distance.append(((LA.norm(np.array(data[0])-np.array(feature))),data[1]))
            for i in ran:
                nearest = []
                distance.sort(key = operator.itemgetter(0))
                #print("Now k is ",i)
                for j in range(0,i):
                    next = distance[j]
                    #print("distance is ",next[0])
                    #print(next[1])
                    nearest.append(next[1])
                total[i]+=1
                mostFreq = self.mostFreq(nearest,i)
                #mostFreq = np.bincount(nearest).argmax()
                #print(mostFreq," ",words[-1])
                if mostFreq!=int(words[-1]):
                    error[i]+=1
        for i in ran:
            #print(error[i]," ",total[i])
            dict[i]=error[i]/total[i]

    def mostFreq(self,nearest,k):
        freq = {}
        for num in nearest:
            if num in freq:
                freq[num]+=1
            else:
                freq[num]=1
        sortedFreq = sorted(freq.items(),key=operator.itemgetter(1),reverse=True)
        current = []
        highest = sortedFreq[0][1]
        for num in sortedFreq:
            if num[1]==highest:
                current.append(num[0])
            else:
                break
        return random.choice(current)

    def calculate(self,data,table):
        for value in data:
            input = []
            for pro in self.projection_data:
                input.append(np.inner(np.array(pro),np.array(value[0])))
            table.append((input,value[1]))

    def trainingDist(self):
        file = open("pa1train.txt","r")
        self.calDist(file,self.training_error_2)
        print(self.training_error_2)

    def validationDist(self):
        file = open("pa1validate.txt")
        self.calDist(file,self.validation_error_2)
        print(self.validation_error_2)

    def testDist(self):
        file = open("pa1test.txt","r")
        self.calDist(file,self.test_error_2)
        print(self.test_error_2)

    def calDist(self,file,dict):
        ran = [1,3,5,9,15]
        total = {1:0,3:0,5:0,9:0,15:0}
        error = {1:0,3:0,5:0,9:0,15:0}
        lines = file.readlines()
        for line in lines:
            words = line.split()
            pre = []
            feature = []
            for word in words[0:len(words)-1]:
                pre.append(int(word))
            for pro in self.projection_data:
                feature.append(np.inner(np.array(pro),np.array(pre)))
            #print(feature)
            distance = []
            for data in self.new_training_data:
                distance.append(((LA.norm(np.array(data[0])-np.array(feature))),data[1]))
            for i in ran:
                nearest = []
                distance.sort(key = operator.itemgetter(0))
                #print("Now k is ",i)
                for j in range(0,i):
                    next = distance[j]
                    #print("distance is ",next[0])
                    #print(next[1])
                    nearest.append(next[1])
                total[i]+=1
                mostFreq = self.mostFreq(nearest,i)
                #mostFreq = np.bincount(nearest).argmax()
                #print(mostFreq," ",words[-1])
                if mostFreq!=int(words[-1]):
                    error[i]+=1
        for i in ran:
            #print(error[i]," ",total[i])
            dict[i]=error[i]/total[i]


if __name__ == '__main__':
    data = Data()

