# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 22:39:01 2020

@author: Carlos Daniel Marin
         Andres Mosquera
         
Universidad del Valle sede Tuluá
"""

#Tratamiento de los datos
import numpy as np
import pandas as pd
import statsmodels.api as sm

#Graficos
import matplotlib.pyplot as plt

#Preprocesamiento de los datos
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics as skM

#Metricas del modelo
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import sklearn.metrics as skM
from sklearn.svm import SVC


#confi warnings

import warnings
warnings.filterwarnings('once')

#Se lee el archivo csv con la base de conocimiento
datos = pd.read_csv('base_datos_py.csv', encoding = "ISO-8859-1", engine='python')
df=pd.DataFrame(datos)

#Se crea un dataframe que contiene los datos de los estudiantes que desertaron
y = df['DesertoNum']


#Se pasan los datos categoricos a one-hot-encoder
codigoDummy = pd.get_dummies(df['Codigo de programa'])
tipoProgramaDummy = pd.get_dummies(df['Tipo de programa'])
jornadaDummy = pd.get_dummies(df['Jornada'])
sexoDummy = pd.get_dummies(df['Sexo'])
ciudadDummy = pd.get_dummies(df['Ciudad residencia'])
zonaDummy = pd.get_dummies(df['Tipo de zona'])
graduadoDummy = pd.get_dummies(df['Graduado'])

#Se dropean los datos del archivo csv innecesarios
df = df.drop(['Tipo de programaNum'], axis='columns')
df = df.drop(['JornadaNum'], axis='columns')
df = df.drop(['SexoNum'], axis='columns')
df = df.drop(['Ciudad residenciaNum'], axis='columns')
df = df.drop(['Tipo de zonaNum'], axis='columns')
df = df.drop(['GraduadoNum'], axis='columns')
df = df.drop(['DesertoNum'], axis='columns')
df = df.drop(['Deserto'], axis='columns')
df = df.drop(['Tipo de programa'], axis='columns')
df = df.drop(['Jornada'], axis='columns')
df = df.drop(['Sexo'], axis='columns')
df = df.drop(['Ciudad residencia'], axis='columns')
df = df.drop(['Tipo de zona'], axis='columns')
df = df.drop(['Graduado'], axis='columns')
df = df.drop(['ID'], axis='columns')
df = df.drop(['Creditos del programa'], axis='columns')
df = df.drop(['Anno de admision'], axis='columns')
df = df.drop(['Numero periodos matriculados'], axis='columns')
df = df.drop(['Asignaturas por estudiante'], axis='columns')
df = df.drop(['Cantidad de creditos aprobados'], axis='columns')
df = df.drop(['Cantidad de creditos matriculados'], axis='columns')
df = df.drop(['Numero periodos por estudiante'], axis='columns')
df = df.drop(['Codigo de programa'], axis='columns')

#Se construye el data set final
datasetFinal = pd.concat([df, codigoDummy, tipoProgramaDummy, jornadaDummy, sexoDummy,
                          ciudadDummy, zonaDummy, graduadoDummy], axis='columns')


#Split de los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datasetFinal, y,
                                                    test_size = 0.25,
                                                    random_state = 42,
                                                    stratify = y)

#Algoritmo de decision
algoritmo = DecisionTreeClassifier(criterion='entropy', splitter="best", max_depth=10)

#Entrenamiento del modelo utilizado
algoritmo.fit(X_train, y_train)

#Algoritmo de predicción
y_pred = algoritmo.predict(X_test)


#Se pasan los datos al SVM
svm = SVC(kernel='linear', )
svm.fit(X_train, y_train)
predic = svm.predict(X_test)


print("********Árbol de decisión***********")
matriz = confusion_matrix(y_test, y_pred)
print("Matriz de confusion")
print(matriz)
print()

#Relacion entre las predicciones correctas y el numero total de predicciones
#Con que frecuencia es correcto el clasificador
exactitud = accuracy_score(y_test, y_pred)
print("Exactitud del modelo") #vp+vn/vp+fp+fn+vn
print(exactitud)
print()

#Relacion entre las predicciones correctas y el numero total de predicciones
#correctas previstas, mide la precision del modelo a la hora de predecir casos positivos
precision = precision_score(y_test, y_pred)
print("Presicion del modelo")
print(precision)
print()

#Relacion entre las predicciones positivas correctas y el numero total de predicciones
#positivas o cuan sensible es el modelo para detectar instancias positivas.
sensibilidad = recall_score(y_test, y_pred) #vp/vp+fn
print("Sensibilidad del modelo") # vp/vp+fn
print(sensibilidad)
print()

#El puntaje F1 es la medida armonica de la memoria y la precision,
#con un puntuacion mas alta, mejor es el modelo.
puntaje = f1_score(y_test, y_pred)
print("medida armonica del modelo")
print(puntaje)
print()

print("Curva ROC")
fpr, tpr, _ = skM.roc_curve(y_test,  y_pred)
auc = skM.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show() 

print("\n")
print("**************SVM******************")
matriz = confusion_matrix(y_test, predic)
print("Matriz de confusion")
print(matriz)
print()

#Relacion entre las predicciones correctas y el numero total de predicciones
#Con que frecuencia es correcto el clasificador
exactitud = accuracy_score(y_test, predic)
print("Exactitud del modelo") #vp+vn/vp+fp+fn+vn
print(exactitud)
print()

#Relacion entre las predicciones correctas y el numero total de predicciones
#correctas previstas, mide la precision del modelo a la hora de predecir casos positivos
precision = precision_score(y_test, predic)
print("Presicion del modelo")
print(precision)
print()

#Relacion entre las predicciones positivas correctas y el numero total de predicciones
#positivas o cuan sensible es el modelo para detectar instancias positivas.
sensibilidad = recall_score(y_test, predic) #vp/vp+fn
print("Sensibilidad del modelo") # vp/vp+fn
print(sensibilidad)
print()

#El puntaje F1 es la medida armonica de la memoria y la precision,
#con un puntuacion mas alta, mejor es el modelo.
puntaje = f1_score(y_test, predic)
print("medida armonica del modelo")
print(puntaje)
print()


print("Curvas ROC")
aucAr = skM.roc_auc_score(y_test, y_pred)
aucSVM = skM.roc_auc_score(y_test, predic)
print("Arboles "+str(aucAr))
print("SVM "+str(aucSVM))

a_fpr, a_tpr, _ = skM.roc_curve(y_test,  y_pred)
svm_fpr, svm_tpr, _ = skM.roc_curve(y_test, predic)

plt.plot(a_fpr,a_tpr,label='auc Árbol de decisión= %0.3f' % aucAr)
plt.plot(svm_fpr,a_tpr,label='auc SVM= %0.3f' % aucSVM)

plt.title('Curva ROC')
plt.legend(loc=4)
plt.xlabel('1-especificidad')
plt.ylabel('Sensibilidad')
plt.show()

