import numpy as np
npa = np.array

class hidden_layer:
    def __init(self,hiddensize,featuresize):
        r  = sqrt(6) / sqrt(hiddenSize+2)
        self.weight= np.random.rand([hiddensize,2*featuresize]); # because inputs are 2
        self.weight=np.multiply(self.weight,2*r)-r
        self.softmax_weights= np.random.rand([5,hiddensize]); # because outputs are 5
        #self.bias=np.random.rand([1,1]);

    def forwardpass(self,x):
        output=np.tanh(np.dot(self.weight,x));
        return output

    def diff(self,x):
        diff=np.power(np.tanh(x),2)
        diff = np.ones(diff.shape)-diff
        return diff

    def softmax(self,w, t = 1.0):
        e = np.exp(npa(w) / t)
        dist = e / np.sum(e)
    return dist


    def softmax_predict(self,x,phase):
        o=np.argmax(self.softmax(np.dot(self.softmax_weights,x)))
        if phase=="train" :




    def softmax_error(self,subtree,ref,label,predict):
        temp1=np.dot(np.transpose(self.softmax_weights),(label-predict))
        se=temp1*diff(ref[subtree['op']])
        return se

    def forward_sentence(self,dict_tree,rep):
        ipm=dict();
        for subtree in dict_tree:
            ip1=rep[subtree['ip1']];
            ip2=rep[subtree['ip2']];
            ip=np.vstack((ip1,ip2));
            ipm[subtree['op']]=ip;
            rep[subtree['op']]=self.forwardpass(ip);

        return ipm

    def backward_sentence(self,dict_tree,rep,ipm,label,predict):
        k=list();
        dele=dict();
        delp=dict()
        k=dict_tree.reverse();
        wgrad=np.zeros(self.weight.shape);
        for subtree in k:
            err_soft=self.softmax_error(subtree,ref,label,predict)
            if subtree['op'] in delp.keys():
                err=err_soft+delp[subtree['op']]
            else:
                err=err_soft;
            err[subtree['op']]=err;
            wgrad=wgrad+ np.dot(err,np.transpose(ipm[subtree['op']]);

            if subtree['has_kids']:
                edown=np.dot(np.transpose(self.weights),err)*diff(ipm[subtree['op']]);
                delp[subtree['left_kid']]=edown[:300,1];
                delp[subtree['right_kid']]=edown[300:,1];

        self.weight=self.weight+0.001*wgrad;





















