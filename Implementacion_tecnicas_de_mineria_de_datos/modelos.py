# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 22:39:01 2020

@author: Carlos Daniel Marin
         Andres Mosquera
         
Universidad del Valle sede Tuluá
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as skM
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns; sns.set(font_scale=1.2)

#DataFrame
datos = pd.read_csv('base_datos.csv', encoding = "ISO-8859-1", engine='python')
df=pd.DataFrame(datos)

#se asignan los valores de cada campo
#ident=df['ID'].values
codigo=df['Codigo de programa'].values
tipoProg=df['Tipo de programa'].values
jornada=df['Jornada'].values
periodoAd=df['Periodo de admision'].values
sexo=df['Sexo'].values
edadIng=df['Edad de ingreso'].values
bra=df['Situacion de BRA'].values
excep=df['Condicion de excepcion'].values
residenc=df['Ciudad residencia'].values
zona=df['Tipo de zona'].values
matriculasxper=df['Numero de asignaturas matriculadas por periodo academico'].values
promedioGen=df['Promedio general del estudiante'].values
promedioCred=df['Promedio de creditos matriculados por periodo'].values
credAprob=df['Proporcion creditos matriculado por aprobados'].values
grado=df['Graduado'].values
y=df['Deserto'].values

#Construcción del dataset de acuerdo al dataframe
X=np.array(list(zip(codigo,tipoProg,jornada,periodoAd,sexo,edadIng,bra,excep,residenc,zona,matriculasxper,promedioGen,promedioCred,credAprob,grado)))

#Split de los datos de entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = 42,
                                                    stratify = y)

#Se pasan los datos al SVM
svm = SVC(kernel='linear', )
svm.fit(X_train, y_train)
#print(svm.score(X_test,y_test))

#Se pasan los datos al Árbol de decisión
algo = DecisionTreeClassifier(criterion='entropy', splitter="best", max_depth=10)
algo.fit(X_train, y_train)

#Se hacen las respectivas pruebas
#SVM
predic = svm.predict(X_test)

#Árbol de decisión
predicAr = algo.predict(X_test)

print("********Árbol de decisión***********")
matriz = skM.confusion_matrix(y_test,predicAr)
print("Matriz:")
print(matriz)

print("Presicion:")
precision = skM.precision_score(y_test,predicAr)
print(precision)

print("Recall score:")
recall = skM.recall_score(y_test,predicAr)
print(recall)

print("F1 score")
f1Score = skM.f1_score(y_test,predicAr)
print(f1Score)

print("Accuracy")
acurazi = skM.accuracy_score(y_test,predicAr)
print(acurazi)


print("\n")
print("**************SVM******************")
matriz = skM.confusion_matrix(y_test,predic)
print("Matriz:")
print(matriz)

print("Presicion:")
precision = skM.precision_score(y_test,predic)
print(precision)

print("Recall score:")
recall = skM.recall_score(y_test,predic)
print(recall)

print("F1 score")
f1Score = skM.f1_score(y_test,predic)
print(f1Score)

print("Accuracy")
acurazi = skM.accuracy_score(y_test,predic)
print(acurazi)

print("\nCurvas ROC")
aucAr = skM.roc_auc_score(y_test, predicAr)
aucSVM = skM.roc_auc_score(y_test, predic)
print("Arboles "+str(aucAr))
print("SVM "+str(aucSVM))

a_fpr, a_tpr, _ = skM.roc_curve(y_test,  predicAr)
svm_fpr, svm_tpr, _ = skM.roc_curve(y_test, predic)

plt.plot(a_fpr,a_tpr,label='auc Árbol de decisión= %0.3f' % aucAr)
plt.plot(svm_fpr,a_tpr,label='auc SVM= %0.3f' % aucSVM)

plt.title('Curva ROC')
plt.legend(loc=4)
plt.xlabel('1-especificidad')
plt.ylabel('Sensibilidad')
plt.show()
