

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 01:15:09 2018

@author: emara
"""
import random 
from scipy import misc
import numpy as np
import string as str
from operator import itemgetter, attrgetter   
from matplotlib import pyplot as plt 
from skimage.io import imread
import scipy

from scipy import stats


        
        

def imgintovecs(start,end,count):
    X = np.zeros([26*count,144])
    chars=[]
    for char in str.ascii_lowercase:
        chars.append(char)
    i=0
    for letter in range(0,26):
        letter=chars[letter]
        for index in range(start,end):
                x="A1%s%d.jpg"%(letter,index)
                img = imread(x)
                img=img.flatten()
                X[i,:] = img 
                i=i+1
    return X


def labeling(a,count):
    chars=[]
    labels=[]
    index=[]


    for char in str.ascii_lowercase:
            chars.append(char)
    for i in range(0,26):
            for k in range(0,count):
                labels.append(ord(chars[i]))
                index.append(k)
    b=np.array(labels)
    

    c =np.column_stack((a,b))
    labeled =np.column_stack((c,index))
	
    
    return labeled

    

        
def spliting(data):
    indecies = list(range(0, 182))
    trainIndecies=random.sample(indecies, 145)
              
    traindata=np.zeros([145,146])
    for i in range(0,145):
        traindata[i]=data[trainIndecies[i]]
    
    testIndecies=[x for x in indecies if x not in trainIndecies]
    
    testdata=np.zeros([37,146])
    for i in range(0,37):
        testdata[i]=data[testIndecies[i]]
    
    return testdata,traindata


def euclideanDistance(instance1, instance2, length):

    distance = 0
    
    for x in range(length):
    
        distance += np.square((instance1[x] - instance2[x]))
    
    return np.sqrt(distance)

def MissOrHit(vector, trainning, k, dismatrix):
    hit=0
    miss=0
    beingtested=vector[-2]   #class of the img being tested
    
    identityDic=[]
    for i in range(0,len(trainning)):
        dis=get_from_the_dismatrix(vector, trainning[i], dismatrix)
        identityDic.append(trainning[i][-2])
        identityDic.append(dis)
    tuples=zip(*[identityDic[x::2] for x in (0, 1)])

    sortedDis=sorted(tuples, key=itemgetter(1))
    
    knearest=[]
    for i in range(k):
        knearest.append(sortedDis[i][0])
    mostfrequent=stats.mode(knearest)
    
#    x=int(mostfrequent[0])
#    print  'beingtested= ',int(beingtested), 'most freq = ', x
    if mostfrequent[0]==beingtested:
        return 1
    else:
        return 0
    
       
    
def get_from_the_dismatrix(vec1, vec2, dismatrix):
    l=len(dismatrix)/26
    vec1type=vec1[-2]
    vec1index=vec1[-1]
    vec1charindex= vec1type - 97
    overcolumn=int((l*vec1charindex)+vec1index)
   
    vec2type=vec2[-2]
    vec2index=vec2[-1]
    vec2charindex=vec2type - 97
    overrow=int((l*vec2charindex)+vec2index)
   
    return dismatrix.item(( overcolumn, overrow))

def allEucl(data):
    l=len(data)
    dis=np.zeros([l,l])
    for i in range(0,l):
        for j in range(0,l):
            val=sum(np.square(data[i]-data[j]))
            dis[i,j]=np.sqrt(val)
    
    return dis


def euclidean4(data):
    disx=np.zeros([182,182])
    for i in range(0,182):
        for j in range(0,182):
            
            vecoutter=np.array(data[i])
            vecinner=np.array(data[j])
            distance = 0
    
        for x in range(0,144):
    
            distance += np.square((vecinner - vecoutter))
    
   
    disx[i,j] = np.sqrt(distance)
    return disx
#   

    



#TestSample, TrainSample=spliting(labeledData)[0],spliting(labeledData)[1]

#rates=[]
#
#for k in range(1,101):
#    error=0.0
#
#    for i in range(len(TestSample)):
#        error += MissOrHit(TestSample[i], TrainSample, k, DistanceMatrix)
#        
#    Rate=(error/37.0) * 100.0
#    rates.append(Rate)
#
#Ks=range(1,101)
#
#plt.plot(Ks,rates)
#plt.rcParams["figure.figsize"] = (4,4)
#plt.xlim(-5, 20)
#plt.ylim(-1, 100)  
#plt.xticks(Ks[0:10])
#plt.xlabel('Values of K', fontsize=12)
#plt.ylabel('one fold Error rate', fontsize=12)
#plt.suptitle('KNN', fontsize=14)
#r=min(rates)
#print 'min error = ',min(rates)
#K_index=(x for x in reversed([y for y in enumerate(rates)]) if x[1] == r).next()[0]
#print 'K= ', Ks[rates.index(r)]
##plt.savefig('KNN.jpg')

'''10 fold 


total_rates=[0]*100

for j in range(0,10):
    TestSample, TrainSample=spliting(labeledData)[0],spliting(labeledData)[1]
    rates=[]

    for k in range(1,101):
        error=0.0

        for i in range(len(TestSample)):
            error += MissOrHit(TestSample[i], TrainSample, k, DistanceMatrix)
            
        Rate=(error/37.0) * 100.0
        rates.append(Rate)
    
    total_rates=[sum(x) for x in zip(rates, total_rates)]

for i in range(0,100):
    total_rates[i]=float(total_rates[i]/(j+1))

Ks=range(1,101)
r=min(total_rates)

print 'min error = ',min(total_rates)
K_index=(x for x in reversed([y for y in enumerate(total_rates)]) if x[1] == r).next()[0]
print 'K= ', Ks[total_rates.index(r)]

plt.plot(Ks,total_rates)

plt.rcParams["figure.figsize"] = (6,4)
plt.xlim(-5, 20)
plt.ylim(-1, 100)  
plt.xticks(Ks[0:20])
plt.xlabel('Values of K', fontsize=12)
plt.ylabel('10-fold Error rate', fontsize=12)
plt.suptitle('KNN', fontsize=14)
#plt.savefig('KNN.jpg')

plt.show()
'''





alldata=imgintovecs(1,9,9)
disMatrix=allEucl(alldata)

Train=imgintovecs(1,7,7)
Test=imgintovecs(7,9,2)

TestLablel=labeling(Test,2)
Trainlabel=labeling(Train,7)


HITS=[0]*26

for i in range(0,len(HITS)):
    for j in range(2):
        HITS[i]+=MissOrHit(TestLablel[i], Trainlabel, 4, disMatrix)
    
            




chars=[]
for char in str.ascii_lowercase:
        chars.append(char)
        

xticks = chars
plt.ylim(0, 3) 
x=range(0,26)
plt.xticks(x, xticks)
plt.scatter(range(0,26),HITS)

plt.suptitle('Accuracy', fontsize=14)
plt.savefig('Accuracy.jpg')
plt.show()



