import numpy as np

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


npa = np.array

class softmax:
    def __init__(self,hiddensize):
        self.softmax_weights= np.random.rand(2,hiddensize);  #because outputs are 5
        r  = math.sqrt(6) / math.sqrt(hiddensize+2)
        self.softmax_weights=np.multiply(self.softmax_weights,2*r)-r


    def softmax(self,w, t = 1.0):
        e = np.exp(npa(w) / t)
        dist = e / np.sum(e)
        return dist

    def softmax_diff(self,x):
        return np.log(self.softmax(np.dot(self.softmax_weights,x)))

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

def softmax_train(xtrain,xlabel):
    model=softmax(3298)
    trainid=(range(len(xtrain)))
    for i in range(500):
        random.shuffle(trainid)
        for j in trainid:
            predict_label,prob=model.softmax_predict(xtrain[j]);
             #print predict_label
            model.softmax_error(xlabel[j],prob,xtrain[j])
    return model

def softmax_test(ytrain,model):
    yout=list();
    for j in ytrain:
        out,mat=model.softmax_predict(j)
        yout.append(out)
    return yout


def create_bag_of_centroids( wordlist, word_centroid_map ):

     #The number of clusters is equal to the highest cluster index

     #in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1

     #Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

     #Loop over the words in the review. If the word is in the vocabulary,
     #find which cluster it belongs to, and increment that cluster count
     #by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

     #Return the "bag of centroids"
    return bag_of_centroids



if __name__ == '__main__':


     model = Word2Vec.load("300features_40minwords_10context")


     # ****** Run k-means on the word vectors and print a few clusters
     #

     start = time.time() # Start time

     # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
     # average of 5 words per cluster
     word_vectors = model.syn0
     print "shape..........."
     print word_vectors.shape
     num_clusters = word_vectors.shape[0] / 5

     # Initalize a k-means object and use it to extract centroids
     print "Running K means"
     kmeans_clustering = KMeans( n_clusters = num_clusters )
     idx = kmeans_clustering.fit_predict( word_vectors )

     # Get the end time and print how long the process took
     end = time.time()
     elapsed = end - start
     print "Time taken for K Means clustering: ", elapsed, "seconds."


     # Create a Word / Index dictionary, mapping each vocabulary word to
     # a cluster number
     word_centroid_map = dict(zip( model.index2word, idx ))

     # Print the first ten clusters
     for cluster in xrange(0,10):
         #
         # Print the cluster number
         print "\nCluster %d" % cluster
         #
         # Find all of the words for that cluster number, and print them out
         words = []
         for i in xrange(0,len(word_centroid_map.values())):
             if( word_centroid_map.values()[i] == cluster ):
                 words.append(word_centroid_map.keys()[i])
         print words


     # Create clean_train_reviews and clean_test_reviews as we did before
     #

     # Read data from files
     train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
     test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )


     print "Cleaning training reviews"
     clean_train_reviews = []
     for review in train["review"]:
         clean_train_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, \
             remove_stopwords=True ))

     print "Cleaning test reviews"
     clean_test_reviews = []
     for review in test["review"]:
         clean_test_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, \
             remove_stopwords=True ))


     # ****** Create bags of centroids
     #
     # Pre-allocate an array for the training set bags of centroids (for speed)
     train_centroids = np.zeros( (train["review"].size, num_clusters), \
         dtype="float32" )

     # Transform the training set reviews into bags of centroids
     counter = 0
     for review in clean_train_reviews:
         train_centroids[counter] = create_bag_of_centroids( review, \
             word_centroid_map )
         counter += 1

     # Repeat for test reviews
     test_centroids = np.zeros((test["review"].size, num_clusters), \
         dtype="float32" )

     counter = 0
     for review in clean_test_reviews:
         test_centroids[counter] = create_bag_of_centroids( review, \
             word_centroid_map )
         counter += 1


     x_train, x_test, y_train, y_test = train_test_split(train_centroids, train["sentiment"], test_size=0.33, random_state=42)
     import pickle
     import pdb

     with open("objs.pickle","w") as f:
        pickle.dump([x_train, x_test, y_train, y_test],f)
    #with open("objs.pickle") as f:
    #    x_train, x_test, y_train, y_test = pickle.load(f)
    #    print len(x_train[0])
    #    print type(x_train)
    #    print len(x_train)
        pdb.set_trace()


     mod = softmax_train(x_train, y_train)
     f1 = open("model_weights.pickle", "w")
     pickle.dump(mod.softmax_weights, f1)
     yout = softmax_test(x_test, mod)
     pdb.set_trace()

     import sklearn.metrics
     accuracy = sklearn.metrics.accuracy_score(y_test, yout)
     print accuracy





