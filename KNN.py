

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
import collections

from scipy import stats


        
        

def imgintovecs(path):
    X = np.zeros([26*7,144])
    chars=[]
    for char in str.ascii_lowercase:
        chars.append(char)
    i=0
    for letter in range(0,26):
        letter=chars[letter]
        for index in range(1,8):
                x="%sA1%s%d.jpg"%(path,letter,index)
                img = imread(x)
                img=img.flatten()
                X[i,:] = img 
                i=i+1
    return X


def labeling(a):
    chars=[]
    labels=[]
    index=[]


    for char in str.ascii_lowercase:
            chars.append(char)
    for i in range(0,26):
            for k in range(0,7):
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

    beingtested=vector[-2]   #class of the img being tested
    
    identityDic=[]
    for i in range(0,len(trainning)):
        dis=get_from_the_dismatrix(vector, trainning[i], dismatrix)
        identityDic.append(trainning[i][-2])
        identityDic.append(dis)
    tuples=zip(*[identityDic[x::2] for x in (0, 1)])
    sortedDis=sorted(tuples, key=itemgetter(1))
    
    knearest={}
    for i in range(k):
        knearest[sortedDis[i][1]]=sortedDis[i][0]
    sknearest=collections.OrderedDict(sorted(knearest.items()))
    
    mostfrequent=stats.mode(sknearest.values())
    if mostfrequent[0]!=beingtested:
        return 1
    else:
        return 0
        
    
def get_from_the_dismatrix(vec1, vec2, dismatrix):
    
    vec1type=vec1[-2]
    vec1index=vec1[-1]
    vec1charindex= vec1type - 97
    overcolumn=int((7*vec1charindex)+vec1index)
   
    vec2type=vec2[-2]
    vec2index=vec2[-1]
    vec2charindex=vec2type - 97
    overrow=int((7*vec2charindex)+vec2index)
    return dismatrix.item(( overcolumn, overrow))

def allEucl(data):
    dis=np.zeros([182,182])
    for i in range(0,182):
        for j in range(0,182):
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

    

aldata=imgintovecs('')
DistanceMatrix=allEucl(aldata)
labeledData=labeling(aldata) 
#
#TestSample, TrainSample=spliting(labeledData)[0],spliting(labeledData)[1]
#
#rates=[]
#
#final={}
#for k in range(1,101):
#    error=0.0
#    
#    for i in range(len(TestSample)):
#        error += MissOrHit(TestSample[i], TrainSample, k, DistanceMatrix)
#    final[k]=error
#
#
#print  min(final, key = final.get)

#   Rate=(error/37.0) * 100.0
#  rates.append(Rate)
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
#plt.savefig('KNN.jpg')

'''10 fold'''


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
plt.xlim(0, 100)
plt.ylim(-1, 100)  
plt.xlabel('Values of K', fontsize=12)
plt.ylabel('10-fold Error rate', fontsize=12)
plt.suptitle('KNN', fontsize=14)
plt.savefig('KNN.jpg')

plt.show()


'''




'''






'''



#def labeling(a):
#    chars=[]
#    labels=[]
#    index=[]
#
#
#    for char in str.ascii_lowercase:
#            chars.append(char)
#    for i in range(0,26):
#            for k in range(0,7):
#                labels.append(ord(chars[i]))
#                index.append(k)
#    b=np.array(labels)
#    
#
#    c =np.column_stack((a,b))
#    labeled =np.column_stack((c,index))
#	
#    
#    return labeled











#np.asarray().reshape(-1)
#np.asarray(testdata).reshape(-1)





























#alldis=allEucl(aldata) 
#numberofrepatrioning=10
#for n in range(0,10):
#    count=0
#    for k in range(1,100):
#        altrain=spliting(aldata)[1] 
#        altest=spliting(aldata)[0]
#        nearest_neighbors=k_nearest_neighbors(altrain, altest, k)
#        percentage = checkAccuracy(nearest_neighbors, altrain, altest)
#        count += 1
#        print ("\nTest set row[%d]:" % count)
#        print ("K value: %d" % k)
#        print ("Accuracy: %d %%" % percentage) 
#   
#        
#        





        




  
    


#n=0
#j=0
#while j<(182):
#    s=X[j:j+7]
#    labeled[chars[n]]=X[j:j+7]
#    j=j+7
#    n=n+1





#dis=np.zeros([182,182])
#for i in range(0,182):
#    for j in range(0,182):
#        sum=0
#        for k in range(0,144):
#             sum+=np.square(X[i, k]-X[j, k])
#        dis[i,j]=np.sqrt(sum)



#
#def splitdata(data):
#    trainindex=[]
#    testindix=[]
#    while len(testindix)!= 37:
#        x=randint(0,181)
#        if x not in trainindex:
#            testindix.append(x)
#    
#        for i in range(0,182):
#            if i not in testindix:
#                trainindex.append(i)
#        traindata=np.zeros([145,144])
#        
#    for i in range(0,135):
#        traindata[i]=data[trainindex[i]]
#    testdata=np.zeros([37,144])
#    for i in range(0,37):
#        testdata[i]=data[testindix[i]]
#    return traindata, testdata
#
#
#
#
#




   
#
#for i in range(0,135):
#    train[i]=X[randint(0,181)]
    
        



            
     
        
#       #trainM= np.zeros([26*5,144])
#testM= np.zeros([26*2,144])
#train=[]
#test=[]
#j=0
#n=0
#
#while j<(182):
#    s=X[j:j+5]
#    train.append(s)
#    j=j+7
#    n=n+1
#j=0
#n=0
#while j<(182):
#    s=X[j+5:j+7]
#    test.append(s)
#    j=j+7
#    n=n+1
#          
#        
#    

#        
#        

#
#       
#    
    
#
#def eucl(point,training):
#    dis=[]
#    sum=0
#    for i in range(0,135):
#        vec_sub=point-training[i]
#        for i in range(0,len(vec_sub)):
#            sum +=np.square(vec_sub[i])
#        dis.append(np.sqrt(sum))
#    return sorted(dis)
#
#diss=eucl(tsd[0],trd)    
#def euclidean4(data):
#    disx=np.zeros([182,182])
#    for i in range(0,182):
#        for j in range(0,182):
#            
#            vecoutter=np.array(data[i])
#            vecinner=np.array(data[j])
#            disx[i,j] = distance.euclidean(vecoutter, vecinner)
#            #np.linalg.norm(vecoutter, vecinner)
#    return disx
##    
#
#
#
#zeeby=euclidean4(X)
   '''