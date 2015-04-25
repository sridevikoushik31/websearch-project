from parse_stuff import *
import numpy as np
npa = np.array
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.cross_validation import train_test_split
from gensim.models import Word2Vec
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
import math
import random
import pdb

class hidden_layer:
    def __init__(self,hiddensize,featuresize,randominit,weight,sweight):
        r  = math.sqrt(6) / math.sqrt(hiddensize+2)
        if (randominit==0):
            self.weight=weight
            self.softmax_weight=sweight
        else:
            self.weight= np.random.rand(hiddensize,2*featuresize); # because inputs are 2
            self.softmax_weights= np.random.rand(2,hiddensize); # because outputs are 5
            self.softmax_weights=np.multiply(self.softmax_weights,2*r)-r
            self.weight=np.multiply(self.weight,2*r)-r
        #self.bias=np.random.rand([1,1]);

    def single_forwardpass(self,x):
        output=np.tanh(np.dot(self.weight,x));
        return output

    def nn_diff(self,x):
        diff=np.power(np.tanh(x),2)
        diff = np.ones(diff.shape)-diff
        return diff

    def softmax(self,w, t = 1.0):
        e = np.exp(npa(w) / t)
        dist = e / np.sum(e)
        return dist

    def softmax_diff(self,x):
        return np.log(self.softmax(x))

    def softmax_predict(self,x):
        temp=(self.softmax(np.dot(self.softmax_weights,x)))
        return np.argmax(temp),temp;

    def softmax_error(self,ilabel,predict,x):
        label=np.zeros([2,1]);
        label[ilabel]=1;
        temp1=np.dot(np.transpose(self.softmax_weights),(label-predict))
        temp2=self.softmax_diff(x)
        se=temp1*temp2;
        self.softmax_weights=self.softmax_weights-0.05*np.dot(predict,np.transpose(se));
        return se

    def forward_sentence(self,dict_tree,rep):
        ipm=dict();
        for subtree in dict_tree:
            ip1=rep[subtree['ip1']];
            ip2=rep[subtree['ip2']];
            ip=np.vstack((ip1,ip2));
            ipm[subtree['op']]=ip;
            rep[subtree['op']]=self.single_forwardpass(ip);
        return ipm,rep


    def backward_sentence(self,dict_tree,rep,ipm,ilabel,predict):
        k=list();
        delp=dict()
        dict_tree.reverse();
        k=dict_tree
        wgrad=np.zeros(self.weight.shape);
        err_soft=self.softmax_error(label,predict,rep[10000])
        i=0
        for subtree in k:
            print i
            i=i+1
            if subtree['op'] in delp.keys():
                err=delp[subtree['op']]
            else:
                err=err_soft;
            #pdb.set_trace()
            wgrad=wgrad+ np.dot(err,np.transpose(ipm[subtree['op']]));

            edown=np.dot(np.transpose(self.weight),err)*self.nn_diff(ipm[subtree['op']]);
            delp[subtree['ip1']]=edown[:300,0].reshape([300,1]);
            delp[subtree['ip2']]=edown[300:,0].reshape([300,1]);

        self.weight=self.weight-0.05*wgrad;
        return np.sum(err_soft,axis=None)


def nn_train(xtrain,xlabel,wordvec_model):
    model=hidden_layer(300,300,1,[],[]);
    trainid=(range(len(xtrain)))
    for i in range(100):
        random.shuffle(trainid)
        for j in trainid:
            rep=dict();
            dict_tree=dict();
            ipm=dict();
            dict_tree,rep=ret_tree(xtrain[j],rep,wordvec_model);
            ipm,rep=model.forward_sentence(dict_tree,rep);
            predict,prob=model.softmax_predict(rep[10000]);
            print prob
            err_soft=model.backward_sentence(dict_tree,rep,ipm,xlabel[j],prob);
            print err_soft
    return model

def nn_test(ytrain,model,wordvec_model):
    yout=list();
    for j in ytrain:
        rep=dict();
        dict_tree=dict();
        ipm=dict();
        dict_tree,rep=ret_tree(j,rep,wordvec_model);
        ipm,rep=model.forward_sentence(dict_tree,rep);
        out,mat=model.softmax_predict(rep[10000])
        yout.append(out)
    return yout


model = Word2Vec.load("300features_40minwords_10context")
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3 )
test = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3 )

x_train, x_test, y_train, y_test = train_test_split(train["review"],train["sentiment"], test_size=0.33, random_state=42)
for i in range(100):
    nmodel=nn_train(blah,label,model)
    yout=nn_test(ytrain,nmodel,model)
    import sklearn.metrics
    accuracy = sklearn.metrics.accuracy_score(y_test, yout)
    print "Iteration :",i*100," Accuracy :" ,accuracy

















