#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:42:52 2019

@author: xiangshuyang
"""
import numpy as np
# We are going to find a classifier for data 
# problem to solve: 
# min \sum_i \xi_i + lambda ||w||^2 subjected to  y_i(w+i x+i -b) \geq 1-\xi_i 
#The language dual give the equivalent problem: max \sum_i alpha_i -1\2 \sum_i \sum_j y_i alpha_i k(x+i,x_j) y_j alpha_j 

class SVM:
    
    def  __init__(self,datamat,label,C,tol,delta,kerneltype,Max): # This function initialize the data, datamat and label should be readen from the give fil
        self.datamatrix=np.matrix(datamat)
        self.labelmatrix=np.matrix(label).T
        self.C=C
        self.tol=tol
        self.delta=delta
        self.m,self.n=np.shape(self.datamatrix)
        self.kernel=np.matrix(np.zeros((self.m,self.m)))
        for i in range(self.m): # The gauss Kernel is to be defined soon after 
            self.kernel[:,i] = self.Kernel(self.datamatrix,self.datamatrix[i,:],delta,'Guass')
        self.alpha=np.matrix(np.zeros((self.m,1)))
        self.b=0 
        self.Max=Max
                
    
    def Kernel(self,X,mu,delta,kerneltype): 
        m,n=np.shape(X)
        K = np.matrix(np.zeros((m,1)))  # this function defines the guass kernel exp(-||x-mu|\^2/delta^2)
        if kerneltype=='Guass':
            for j in range(m):
                deltaRow = X[j,:] - mu
                K[j] = deltaRow*deltaRow.T
                kernel = np.exp(K /(-1*delta**2))
        elif kerneltype =='linear':
            kernel=X*mu.transpose()
        return kernel
        
        
    def Er(self,i): # the value of \alpha\cdot y kernel + b 
        f = np.multiply(self.alpha,self.labelmatrix).T*(self.datamatrix*self.datamatrix[i,:].T) + self.b
        eri= f-self.labelmatrix[i]           
        return eri
    
    def upEk(self,k):
        Ek=self.E(k)
        
 
   
       
    
    def selectJ(self,i):
        j=i
        erj=0
        while (j==i):
            j = int(random.uniform(0,m))
        erj=self.Er(j)
        return j, erj
   
    def smo_onelevel(self,i):
        eri=self.Er(i)
        if ((self.labelmatrix[i]*eri < -self.tol) and (self.alpha[i] < self.C)) or\
           ((self.labelmatrix[i]*eri > self.tol) and (self.alpha[i] > 0)):
            j,erj = self.selectJ(i)
            alphaiold = self.alpha[i].copy();
            alphajold = self.alpha[j].copy();   
            if (self.labelmatrix[i] != self.labelmatrix[j]):
                 L = max(0, self.alpha[j] - self.alpha[i])
                 H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:
                L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                H = min(self.C, self.alpha[j] + self.alpha[i])
            if L==H:
                print ("L==H")
                return 0 
            eta = 2.0 * self.kernel[i,j] - self.kernel[i,i]-self.kernel[j,j]
            if eta>=0 :
                print("eta>0")
                return 0
            else:
                self.alpha[j] -= self.labelmatrix[j]*(eri - erj)/eta
                self.alpha[j]=np.where(np.where(self.alpha[j]>H,H,self.alpha[j])<L,L,self.alpha[j])
                
                if (abs(self.alpha[j] - alphajold) < 0.00001): 
                    print("j almost does not move")
                    return 0 
                else:
                    self.alpha[i] += self.labelmatrix[j]*self.labelmatrix[i]*(alphajold - self.alpha[j])
                
                    preij1=self.labelmatrix[i]*(self.alpha[i]-alphaiold)*self.kernel[i,i]\
-                         +self.labelmatrix[j]*(self.alpha[j]-alphajold)*\
                                self.kernel[i,j]
                    preij2 = self.labelmatrix[i]*(self.alpha[i]-alphaiold)*self.kernel[i,j]\
-                         +self.labelmatrix[j]*(self.alpha[j]-alphajold)*\
                                self.kernel[j,j]    
                
                    bi=self.b-eri-preij1
                    bj=self.b-erj-preij2
                
                
                    if (0 < self.alpha[i]) and (self.C > self.alpha[i]): 
                        self.b = bi
                    elif (0 < self.alpha[j]) and (self.C > self.alpha[j]):
                        self.b = bj
                    else: 
                        self.b = (bi + bj)/2
                    return 1
        else: 
          return 0                
   
    def smo_whole(self):
        iter = 0
        a=0
        entireSet = True
        alphaPairsChanged = 1
        while (iter < self.Max) and (alphaPairsChanged > 0):
            alphaPairsChanged = 0
            for i in range(self.m):
                a= self.smo_onelevel(i)
                if a!=0:
                   alphaPairsChanged+=1
            iter+=1   
        return self.alpha,self.b,alphaPairsChanged
    
   

   
    
    def plot(self):
        import matplotlib.pyplot as plt
        
        dataArr = np.array(self.datamatrix)
        n = np.shape(dataArr)[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if self.labelmatrix[i]==-1:
               xcord1.append(dataArr[i,1])
               ycord1.append(dataArr[i,2])
            else:
                xcord2.append(dataArr[i,1])
                ycord2.append(dataArr[i,2])
        plt.scatter(xcord1, ycord1, s=40, c='red', alpha=0.5)
        plt.scatter(xcord2, ycord2, s=40, c='blue', alpha=0.5)
       
 ##### test#######
    
dataMat = []; labelMat = []
fr = open('textset-kernel.txt')
for line in fr.readlines():
    lineArr = line.strip().split()
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    labelMat.append((float(lineArr[2])))
g=SVM(dataMat,labelMat,500,0.0001,1.3,'Guass',40) 
alpha,b, alphaPairsChanged=g.smo_whole()

