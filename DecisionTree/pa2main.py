from __future__ import division
from numpy import linalg as LA
from heapq import heappush,heappop,nsmallest
import numpy as np
import operator
import random
import math

class Node():
    def __init__(self,split1, split2, larger, feature1, feature2):
        self.pure = 0
        self.splitC = split1
        self.featureC = feature1
        self.featureN = feature2
        self.splitN = split2
        self.larger = larger
        self.nodes = []
        self.childL = 0
        self.childS = 0
        self.prediction = -1

    def checkPure(self,data):
        if len(self.nodes)==1:
            self.pure = 1
            if (self.nodes[0],0) in data:
                self.prediction = 0
            else:
                self.prediction = 1
            return
        for i in range(0,len(self.nodes)-1):
            if ((self.nodes[i],0) in data) != ((self.nodes[i+1],0) in data):
                self.pure = 0
                return
        self.pure = 1
        if (self.nodes[0],0) in data:
            self.prediction = 0
        else:
            self.prediction = 1

    def printInfo(self):
        print("split point: ",self.splitN,", feature of split: ",(self.featureN+1),", prev L/S: ",self.larger)
        print("number of nodes: ",len(self.nodes))

class Create():
    def __init__(self):
        #self.newfile()
        self.root = Node(0,0,0,0,0)
        self.data = self.train()
        self.root.printInfo()
        self.root.childS.printInfo()
        self.root.childL.printInfo()
        self.root.childS.childS.printInfo()
        self.root.childS.childL.printInfo()
        self.root.childL.childS.printInfo()
        self.root.childL.childL.printInfo()

    def train(self):
        #first read training data
        data = []
        file = open("pa2train.txt","r")
        #file = open("nfile.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            feature = []
            for word in words[0:len(words)-1]:
                feature.append(float(word))
            data.append((feature, float(words[-1])))
            #print ("adding feature", feature)
            self.root.nodes.append(feature)
        self.buildTree(data)
        return data

    def trainingError(self):
        total = 0
        error = 0
        for feature in self.data:
            if feature[1]!=self.predict(feature[0]):
                error += 1
            total += 1
        print(error/total)

    def testingError(self):
        total = 0
        error = 0
        data = []
        file = open("pa2test.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            feature = []
            for word in words[0:len(words)-1]:
                feature.append(float(word))
            data.append((feature, float(words[-1])))
        for feature in data:
            if feature[1]!=self.predict(feature[0]):
                error += 1
            total += 1
        print(error/total)

    def validationError(self):
        total = 0
        error = 0
        data = []
        file = open("pa2validation.txt","r")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            feature = []
            for word in words[0:len(words)-1]:
                feature.append(float(word))
            data.append((feature, float(words[-1])))
        for feature in data:
            if feature[1]!=self.predict(feature[0]):
                error += 1
            total += 1
        return (error/total)

    def prune(self):
        fakeS = Node(self.root.splitN,0,0,self.root.featureN,0)
        fakeS.pure = 1
        fakeS.nodes = self.root.childS.nodes
        num0 = 0
        num1 = 0
        for node in fakeS.nodes:
            if (node,0) in self.data:
                num0 += 1
            else:
                num1 += 1
        print(num0,"  num  ",num1)
        if num0>=num1:
            fakeS.prediction = 0
        else:
            fakeS.prediction = 1
        ve = self.validationError()
        trueS = self.root.childS
        self.root.childS = fakeS
        nve = self.validationError()
        print(ve,"  error  ",nve)
        if nve > ve:
            self.root.childS = trueS
        self.trainingError()
        self.testingError()

        fakeL = Node(self.root.splitN,0,1,self.root.featureN,0)
        fakeL.pure = 1
        fakeL.nodes = self.root.childL.nodes
        num0 = 0
        num1 = 0
        for node in fakeL.nodes:
            if (node,0) in self.data:
                num0 += 1
            else:
                num1 += 1
        print(num0,"  num  ",num1)
        if num0>=num1:
            fakeL.prediction = 0
        else:
            fakeL.prediction = 1
        ve = self.validationError()
        trueL = self.root.childL
        self.root.childL = fakeL
        nve = self.validationError()
        print(ve,"  error  ",nve)
        if nve > ve:
            self.root.childL = trueL
        self.trainingError()
        self.testingError()



    def buildTree(self, data):
        frontier = [self.root]
        #while frontier:
        next = []
        dic = {}
        for i in range(0,22):
            dic[i] = 0
        while frontier:
            for node in frontier:
                if node.pure:
                    continue
                threshold = self.findBest(node.nodes,data)
                #print threshold
                dic[threshold[1]] += 1
                node.featureN = threshold[1]
                node.splitN = threshold[0]
                largeN = Node(threshold[0],0,1,threshold[1],0)
                smallN = Node(threshold[0],0,0,threshold[1],0)
                node.childL = largeN
                node.childS = smallN
                for n in node.nodes:
                    #print(n[threshold[1]]," ",threshold[0])
                    if n[threshold[1]]<threshold[0]:
                        smallN.nodes.append(n)
                        #print("in small")
                    else:
                        largeN.nodes.append(n)
                        #print("in large")
                smallN.checkPure(data)
                largeN.checkPure(data)
                next.append(smallN)
                next.append(largeN)
            frontier = next
            next = []
            #print("end of level")
        print dic

    def findBest(self,nodes,data):
        val = 1000
        point = 0
        feature = 0
        for i in range(0,22):
            #print("search feature ",i)
            arr = []
            for node in nodes:
                arr.append(node[i])
            arr.sort()
            mid = []
            for j in range(0,len(arr)-1):
                if arr[j]!=arr[j+1]:
                    mid.append((arr[j]+arr[j+1])/2)
            #print("mids ",mid)
            bestIG = self.getIG(nodes,i,mid,data)
            #print("best is: ",bestIG[1],"  and the point is ",bestIG[0])
            if bestIG[1]<=val:
                val = bestIG[1]
                point = bestIG[0]
                feature = i
                #print(val, point, feature)

        return (point,feature)

    def getIG(self,nodes,i,mid,data):
        nodes.sort(key=operator.itemgetter(i))
        #print(nodes)
        test = []
        for node in nodes:
            if (node,0) in data:
                test.append((node[i],0))
            else:
                test.append((node[i],1))
        #print  test
        PL = len(nodes)
        L0 = 0
        L1 = 0
        PS = 0
        S0 = 0
        S1 = 0
        val = 1000
        ret = 0
        c = 0
        for j in range(0,len(nodes)):
            if (nodes[j],0) in data:
                L0 += 1
            else:
                L1 += 1
        for j in range(0,len(nodes)-1):
            PS += 1
            PL -= 1
            if (nodes[j],0) in data:
                L0 -= 1
                S0 += 1
            else:
                L1 -= 1
                S1 += 1
            #print(nodes[j][i],"  ",nodes[j+1][i])
            if nodes[j][i]!=nodes[j+1][i]:
                curr = self.calculation(PL,L0,L1,PS,S0,S1)
                #print("[PL,L0,L1,PS,S0,S1,value]",PL,L0,L1,PS,S0,S1,curr)
                if curr<= val:
                    val = curr
                    #print (c," / ",len(mid),"  ", j, " / ",len(nodes),"  ",nodes[j][i], nodes[j+1][i])
                    ret = mid[c]
                    c += 1
        #print(ret,val,i)
        return (ret,val)

    def calculation(self,PL,L0,L1,PS,S0,S1):
        total = PL + PS
        if PL>0:
            if L0 > 0:
                l0 = -(L0/PL)*math.log(L0/PL)
            else:
                l0 = 0
            if L1 > 0:
                l1 = -(L1/PL)*math.log(L1/PL)
            else:
                l1 = 0
            pl = l0+l1
        else:
            pl = 0
        if PS > 0:
            if S0 > 0:
                s0 = -(S0/PS)*math.log(S0/PS)
            else:
                s0 = 0
            if S1 > 0:
                s1 = -(S1/PS)*math.log(S1/PS)
            else:
                s1 = 0
            ps = s0+s1
        else:
            ps = 0
        #print(PL,pl,PS,ps,L0,L1,S0,S1)
        return PL/total*pl + PS/total*ps

    def predict(self,feature):
        curr = self.root
        while curr.pure!=1:
            if feature[curr.featureN]<=curr.splitN:
                curr = curr.childS
            else:
                curr = curr.childL
        return curr.prediction

    def newfile(self):
        file = open("pa2train.txt","r")
        nf = open("nfile.txt","w+")
        lines = file.readlines()
        for line in lines:
            words = line.split()
            if float(words[4]) >= 0.5:
                nf.write(line)
        nf.close()
        print("fin")

if __name__ == '__main__':
    create = Create()
    create.trainingError()
    create.testingError()
    create.prune()
