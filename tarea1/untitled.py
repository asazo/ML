
# coding: utf-8

# # Ejercicio 4

# **(a) Lea los archivos de datos y cárguelos en dos dataframe o matrices X, y. En el caso de X es extremadamente importante que mantenga el formato disperso (sparse) (¿porqué?).**

# In[1]:

import os.path

data_base_path = "./data"


# In[2]:

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread

X_train = csr_matrix(mmread(os.path.join(data_base_path, 'train.x.mm')))
y_train = np.loadtxt(os.path.join(data_base_path, 'train.y.dat'))

X_test = csr_matrix(mmread(os.path.join(data_base_path, 'test.x.mm')))
y_test = np.loadtxt(os.path.join(data_base_path, 'test.y.dat'))


# Se usa una matriz sparse ya que, producto de que hay muchos datos vacios (normal en texto). Así se comprime la matriz y queda en un tamaño manejable.

# **(b) Construya un modelo lineal que obtenga un coeficiente de determinación (sobre el conjunto de pruebas) de al menos 0.75.**

# In[3]:

X_train.get_shape()


# Se observa que hay 1147 peliculas cada una con 145256 atributos.

# In[4]:

#from sklearn.preprocessing import StandardScaler

#x_scaler = StandardScaler(with_mean=False)

#X_train = x_scaler.fit_transform(X_train, y_train)
#X_test = x_scaler.transform(X_test, y_test)


# Se prueba escalar los datos pero esto devuelve peores resultados.

# In[5]:

from sklearn.feature_selection import SelectKBest, chi2

ch2 = SelectKBest(chi2, k=29000)

X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)


# Se utiliza la extracción de caracteristicas KBest mediante la metrica chi cuadrado. Esto ya que documentación de sklearn muestra que es utilizada cuando la data esta basada en texto (como es el caso).
# 
# La cantidad de catacteristicas se obtiene mediante prueba y error.
# 
# Fuente: http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py

# In[9]:
print ":D"

import sklearn.linear_model as lm

model = lm.ElasticNet(alpha = 0.05, tol=0.001)
model.fit(X_train, y_train)

print "R2=%f" % model.score(X_test, y_test)


# In[ ]:



