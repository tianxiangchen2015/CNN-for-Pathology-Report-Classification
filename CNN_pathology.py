
# coding: utf-8

# In[99]:

import numpy as np;
import pandas as pd;

import cPickle
import re,csv
import pprint
pp = pprint.PrettyPrinter(indent = 4)


# In[100]:

import sys
sys.path.append('/ccs/home/tchen21/neon/build/lib')
sys.path.append('/ccs/home/tchen21/ConfigArgParse-0.10.0/build/lib')


# In[113]:

from neon.backends import gen_backend
be = gen_backend(backend='gpu',batch_size=32)
print be

import time
csvName=str(int(time.time()))+'.csv'
# In[114]:

import logging
from neon.callbacks.callbacks import Callbacks
from neon.data import ArrayIterator
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine,Linear
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, Misclassification, MeanSquared, Accuracy,Softmax
from neon.util.argparser import NeonArgparser
from neon.util.compat import PY3, range
from neon.data import ArrayIterator, load_mnist
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, Sequential, MergeMultistream
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.layers import Conv, Affine, Pooling
from neon.initializers import Uniform


# In[115]:

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map and len(x)< max_l+2*pad:
#         if word in word_idx_map and len(x)< max_l:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
#     while len(x) < max_l:
        x.append(0)
    return x


# In[116]:

def make_idx_data_cv(revs, word_idx_map,w, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    eleTy={}
    counter=0
    train, test = [], []
    train_l,test_l=[],[]
    x_train,x_test=[],[]
    for rev in revs:
        counter+=1
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        try:eleTy[len(sent)]+=1
        except KeyError: eleTy.update({(len(sent)):1})
        if rev["split"]==cv:
            test.append(w[sent])
            test_l.append(rev["y"])
        else:
            train.append(w[sent])
            train_l.append(rev["y"])
    #pp.pprint(train)
    '''
    print 'type', type(train)
    eleTy={}
    print 'element types'
    for x in train:matrix(y_true, y_pred[, ...])
        for y in x:
            try:eleTy[str(type(x))]+=1
            except KeyError: eleTy.update({str(type(x)):1})
    #pp.pprint(eleTy)
    try: train = np.asarray(train,dtype="int")
    except ValueError:
            train = np.asarray(train,dtype="object")
            print 'train shape', train.shape
    try: test = np.asarray(test,dtype="int")
    except ValueError:
        test = np.asarray(test,dtype="object")
        print 'test shape', test.shape
    '''
    pp.pprint(eleTy)
    print 'counter',counter
    print 'sent last'
    print sent[-1]
    train = np.asarray(train)
    test = np.asarray(test)
    test_l=np.asarray(test_l)
    train_l=np.asarray(train_l)
    
    return train,train_l,test,test_l


# In[125]:

x = cPickle.load(open("mr12index.g","rb"))


# In[126]:

revs, W, W2, word_idx_map,word_idx_rand, vocab, corpusMaxL, classlist= x[0], x[1], x[2], x[3], x[4],x[5],x[6],x[7]


# In[127]:
print classlist

y_true=[]
y_pred=[]
Macro=[]
Micro=[]

for i in range(10):
    be = gen_backend(backend='gpu',batch_size=32)

    x_train, y_train, x_test, y_test=make_idx_data_cv(revs, word_idx_map, W, i, max_l=1500,k=300, filter_h=5)
    x=[]
    xt=[]
    for ii in x_train:
        x.append(np.concatenate(ii))
    x=np.array(x)

    for ii in x_test:
        xt.append(np.concatenate(ii))
    xt=np.array(xt)

    print xt.shape

    from neon.data import ArrayIterator
    from neon.layers import Conv, Affine, Pooling
    from neon.initializers import Uniform
    train_set=ArrayIterator([x,x,x], y_train, nclass=12,lshape=(1, 1508, 300))
    test_set=ArrayIterator([xt,xt,xt], y_test, nclass=12,lshape=(1, 1508, 300))

    print train_set.ndata
    print train_set.shape

    init_norm = Gaussian(loc=0.0, scale=0.01)
    path1 = Sequential(layers=[Conv(fshape=(3,300,512), init=init_norm, activation=Rectlin()),
                               Pooling(fshape=(512,1))])
    path2 = Sequential(layers=[Conv(fshape=(4,300,512), init=init_norm, activation=Rectlin()),
                               Pooling(fshape=(512,1))])
    path3 = Sequential(layers=[Conv(fshape=(5,300,512), init=init_norm, activation=Rectlin()),
                               Pooling(fshape=(512,1))]) 
    layers = [MergeMultistream(layers=[path1, path2, path3], merge="stack"),
              Affine(nout=12, init=init_norm, activation=Softmax())]

    CNN = Model(layers=layers)
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    optimizer=GradientDescentMomentum(learning_rate=0.005,momentum_coef=0.9)
    callbacks=Callbacks(CNN, eval_set=test_set)
    CNN.fit(train_set, optimizer=optimizer, cost=cost, callbacks=callbacks, num_epochs=180)

    predic = CNN.get_outputs(test_set)
    error=CNN.eval(test_set, metric=Misclassification())*100
    print('Misclassification error = %.1f%%' % error)
    predic.shape
    predic = predic.argmax(axis=1).tolist()

    print y_test
    print predic

    y_true.append(y_test)
    y_pred.append(predic)

yt=np.concatenate(y_true)
y=np.concatenate(y_pred)

np.save(str(int(time.time()))+'pred',y)
np.save(str(int(time.time()))+'true',yt)

with open(csvName, 'wb') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow(['pred','actual'])
    for i in range(len(y_true)):
        outrow = []
        outrow.append(y_pred[i])
        outrow.append(y_true[i])
        writer.writerow(list(outrow))
    writer.writerow(['pred_10','actual_10'])
    y = y.tolist()
    yt = yt.tolist()
    print y
    print yt
    finalRow=[y,yt]
    writer.writerow(finalRow)
