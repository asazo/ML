
# coding: utf-8

# # Pregunta 3

# In[1]:

import cPickle as pickle
import numpy as np
import os

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[2]:

def load_CIFAR_one(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        return X, np.array(Y)

def load_CIFAR10(PATH, n_val=10000):
    if n_val > 10000:
        n_val = 10000
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(PATH, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_one(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xval = Xtr[-n_val:Xtr.shape[0]]
    Yval = Ytr[-n_val:Ytr.shape[0]]
    Xtr = Xtr[:-n_val]
    Ytr = Ytr[:-n_val]
    del X, Y
    Xte, Yte = load_CIFAR_one(os.path.join(PATH, 'test_batch'))
    return Xtr, Ytr, Xval, Yval, Xte, Yte


from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler

def scaler_function(Xtr,Xval,Xt,other=True, scale=True):
    if other:
        Xtr_scaled = Xtr / 255.
        Xval_scaled = Xval / 255.
        Xt_scaled = Xt / 255.
        return Xtr_scaled, Xval_scaled, Xt_scaled
    scaler = StandardScaler(with_std=scale).fit(Xtr)
    Xtr_scaled = scaler.transform(Xtr)
    Xval_scaled = scaler.transform(Xval)
    Xt_scaled = scaler.transform(Xt)
    return Xtr_scaled, Xval_scaled, Xt_scaled


from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model


from top_level_features import hog_features
from top_level_features import color_histogram_hsv
from top_level_features import extract_features

#Xtr, Ytr, Xval, Yval, Xte, Yte = load_CIFAR10('.')

#features_Xtr = extract_features(Xtr,[hog_features, color_histogram_hsv])
#features_Xval = extract_features(Xval,[hog_features, color_histogram_hsv])

#Ytr_cat = to_categorical(Ytr, 10)
#Yval_cat = to_categorical(Yval, 10)
#Yte_cat = to_categorical(Yte, 10)

from sklearn.svm import SVC

Xtr, Ytr, Xval, Yval, Xte, Yte = load_CIFAR10('.')
# Escalar datos
#Xtr, Xval, Xte = scaler_function(Xtr, Xval, Xte, other=False)
features_Xtr = extract_features(Xtr,[hog_features, color_histogram_hsv])
features_Xval = extract_features(Xval,[hog_features, color_histogram_hsv])

Ytr_cat = to_categorical(Ytr, 10)
Yval_cat = to_categorical(Yval, 10)
Yte_cat = to_categorical(Yte, 10)

C = np.logspace(-5, 3, 9)
kernel = ['rbf','poly', 'sigmoid']
print "Features All"
for k in kernel:
    for c in C:
        print "Kernel %s"%k+", C=%.0e"%c
        svm = SVC(kernel=k, C=c, verbose=False)
        "Entrenando..."
        svm.fit(features_Xtr, Ytr)
        "Calculando score..."
        score = svm.score(features_Xval, Yval)
        print "Score sobre conjunto de validación:",score

## Solo features hsv
features_Xtr = extract_features(Xtr,[color_histogram_hsv])
features_Xval = extract_features(Xval,[color_histogram_hsv])
C = np.logspace(-5, 3, 9)
kernel = ['rbf', 'poly', 'sigmoid']
print "Features HSV"
for k in kernel:
    for c in C:
        print "Kernel %s"%k+", C=%.0e"%c
        svm = SVC(kernel=k, C=c, verbose=False)
        "Entrenando..."
        svm.fit(features_Xtr, Ytr)
        "Calculando score..."
        score = svm.score(features_Xval, Yval)
        print "Score sobre conjunto de validación:",score


features_Xtr = extract_features(Xtr,[hog_features])
features_Xval = extract_features(Xval,[hog_features])
C = np.logspace(-5, 3, 9)
kernel = ['rbf', 'poly', 'sigmoid']
print "Features HOG"
for k in kernel:
    for c in C:
        print "Kernel %s"%k+", C=%.0e"%c
        svm = SVC(kernel=k, C=c, verbose=False)
        "Entrenando..."
        svm.fit(features_Xtr, Ytr)
        "Calculando score..."
        score = svm.score(features_Xval, Yval)
        print "Score sobre conjunto de validación:",score
