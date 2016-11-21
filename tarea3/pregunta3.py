
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


# In[3]:

Xtr, Ytr, Xval, Yval, Xte, Yte = load_CIFAR10('.')


# In[4]:

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


# In[5]:

"""%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(1,1))
sample = Xtr[12].reshape(3,32,32).T
plt.imshow(sample, interpolation="nearest")
plt.show()

Xtr_scaled, Xt_scaled = scaler_function(Xtr, Xte, other=True)
plt.figure(figsize=(1,1))
sample = Xtr_scaled[12].reshape(3,32,32).T
plt.imshow(sample, interpolation="nearest")
plt.show()

Xtr_scaled, Xt_scaled = scaler_function(Xtr.astype(np.float64), Xte.astype(np.float64), other=False, scale=False)
plt.figure(figsize=(1,1))
sample = Xtr_scaled[12].reshape(3,32,32).T
plt.imshow(sample, interpolation="nearest")
plt.show()

Xtr_scaled, Xt_scaled = scaler_function(Xtr.astype(np.float64), Xte.astype(np.float64), other=False, scale=True)
plt.figure(figsize=(1,1))
sample = Xtr_scaled[12].reshape(3,32,32).T
plt.imshow(sample, interpolation="nearest")
plt.show()"""


# In[5]:

from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model


# In[6]:

Xtr, Xval, Xte = scaler_function(Xtr, Xval, Xte, other=False)
Ytr_cat = to_categorical(Ytr, 10)
Yval_cat = to_categorical(Yval, 10)
Yte_cat = to_categorical(Yte, 10)


# In[7]:

"""model = Sequential()
model.add(Dense(100, input_dim=Xtr.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, init='uniform', activation='softmax'))
model.compile(optimizer=SGD(lr=0.05), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(Xtr, Ytr_cat, nb_epoch=50, batch_size=32, verbose=1, validation_data=(Xval,Yval_cat))
model.save('arch_1.h5')
del model"""

model = Sequential()
model.add(Dense(200, input_dim=Xtr.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, init='uniform', activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(50, init='uniform', activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, init='uniform', activation='softmax'))
model.compile(optimizer=SGD(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(Xtr, Ytr_cat, nb_epoch=50, batch_size=32, verbose=1, validation_data=(Xval,Yval_cat))
model.save("arch_3.h5")

