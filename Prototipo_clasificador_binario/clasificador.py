# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:37:56 2021

@author: Carlos Daniel Marin
         Andres Mosquera
         
Universidad del Valle sede Tuluá
"""
from tkinter import ttk
import tkinter as tk
import tkinter.font as tkFont

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

#confi warnings
import warnings
warnings.filterwarnings('once')

def botonAcc():
    #Datos
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

    tecElec = 0.0
    tecSis = 0.0
    tecAlim = 0.0
    trabSoci = 0.0
    ingSis = 0.0
    ingAlim = 0.0
    contadur = 0.0
    admin = 0.0
    
    #Variables predicción
    codigoPred= float(comboCodProg.get())
    
    if codigoPred == 2710.0 :
        tecElec = 1.0
        
    if codigoPred == 2711.0 :
        tecSis = 1.0
        
    if codigoPred == 2712.0 :
        tecAlim = 1.0
        
    if codigoPred == 3249.0 :
        trabSoci = 1.0
        
    if codigoPred == 3743.0 :
        ingAlim = 1.0
        
    if codigoPred == 3753.0 :
        ingSis = 1.0
        
    if codigoPred == 3841.0 :
        contadur = 1.0
        
    else: 
        admin = 1.0
    
    if comboTipoProg.get() == "1 (Pregrado)":
        pregra = 1.0
        tecno = 0.0
    else:
        tecno = 1.0
        pregra = 0.0
    
    if comboJornada.get() == "1 (Diurno)":
        diurn = 1.0
        nocturn = 0.0
    else:
        nocturn = 1.0
        diurn = 0.0
        
    if comboPeriodoAdmi.get() == "1 (Primer periodo)":
        periodoAdPred= 1.0
    else:
        periodoAdPred= 2.0
    
    if comboSexo.get() == "1 (Masculino)":
        mascul = 1.0
        femen = 0.0
    else:
        femen = 1.0
        mascul = 0.0
      
    edadIngPred= float(edad.get())
    
    braPred= float(comboBRA.get())
    
    if comboExcepcion.get() == "0 (No tiene excepcion)":
        excepPred= 0.0
    else:
        excepPred= 1.0
        
    if comboCity.get() == "1 (Tuluá)":
        tulua = 1.0
        otro = 0.0
    else:
        otro = 1.0
        tulua = 0.0
    
    if comboTipoZona.get() == "1 (Urbana)":
        urbana = 1.0
        rural = 0.0
    else:
        rural = 1.0 
        urbana = 0.0
        
    matriculasxperPred= float(matrixPer.get())
    
    promedioGenPred= float(promGen.get())
    
    promedioCredPred= float(promCredxPer.get())
    
    credAprobPred= float(credAproba.get())
    
    si = 0.0
    no = 1.0
        
    #Fin datos prediccion
    
    #Árbol de decisión
    #Se pasan los datos al modelo
    algo = DecisionTreeClassifier(criterion='entropy', splitter="best", max_depth=10)
    algo.fit(X_train, y_train)
    
    #datos ingresados sobre el estudiante
    student = np.array([periodoAdPred,edadIngPred,braPred,excepPred,matriculasxperPred,promedioGenPred,promedioCredPred,
                        credAprobPred,tecElec,tecSis,tecAlim,trabSoci,ingSis,ingAlim,contadur,admin,pregra,tecno,diurn,
                        nocturn,femen,mascul,otro,tulua,rural,urbana,no,si])
    
    matrEst = np.array([student,student])
    
    predic = algo.predict(matrEst)
    
    if(predic[1] == 1):
        des = "desertar"
    else:
        des = "no desertar"
        
    fontS = tkFont.Font(family="times new roman", size=12)
    tk.Label(tkVent, text="El estudiante tiene tendencia a: "+des, bg="#FFFFFF", font= fontS).place(x=330,y=460)
    

#Interfaz
tkVent = tk.Tk()
tkVent.geometry("850x550")
tkVent.resizable(0,0)
tkVent.iconbitmap('img/logoU.ico')
tkVent.config(bg="white")
tkVent.title("Clasificador binario")

#Creación de la imagen                
fondo = tk.PhotoImage(file="img/fondo.png")
labelFondo = tk.Label(tkVent,image=fondo).place(x=0, y=0)

imagen = tk.PhotoImage(file="img/logo.png")
labelImg = tk.Label(tkVent,image=imagen, bg="#FFFFFF").place(x=520, y=5)

fontStyle = tkFont.Font(family="times new roman", size=20)
labelTittle = tk.Label(tkVent, text="Clasificador binario de deserción", bg="#FFFFFF", font= fontStyle).place(x=250, y=70)

tk.Label(tkVent, text="Codigo de programa:", bg="#FFFFFF").place(x=10,y=145)
comboCodProg = ttk.Combobox(tkVent, state="readonly")
comboCodProg["values"]=[2710,2711,2712,3249,3743,3753,3841,3845]
comboCodProg.place(x = 190,y =145)

tk.Label(tkVent, text="Tipo de programa:", bg="#FFFFFF").place(x=10,y=175)
comboTipoProg = ttk.Combobox(tkVent, state="readonly")
comboTipoProg["values"]=["1 (Pregrado)","2 (Tecnológico)"]
comboTipoProg.place(x = 190,y =175)

tk.Label(tkVent, text="Jornada:", bg="#FFFFFF").place(x=10,y=205)
comboJornada = ttk.Combobox(tkVent, state="readonly")
comboJornada["values"]=["1 (Diurno)","2 (Nocturno)"]
comboJornada.place(x = 190,y =205)

tk.Label(tkVent, text="Periodo de admisión:", bg="#FFFFFF").place(x=10,y=235)
comboPeriodoAdmi = ttk.Combobox(tkVent, state="readonly")
comboPeriodoAdmi["values"]=["1 (Primer periodo)","2 (Segundo periodo)"]
comboPeriodoAdmi.place(x = 190,y =235)

tk.Label(tkVent, text="Sexo:", bg="#FFFFFF").place(x=10,y=265)
comboSexo = ttk.Combobox(tkVent, state="readonly")
comboSexo["values"]=["1 (Masculino)","2 (Femenino)"]
comboSexo.place(x = 190,y =265)

tk.Label(tkVent, text="Edad de ingreso:", bg="#FFFFFF").place(x=10,y=295)
edad = tk.Entry(tkVent,  width=6)
edad.place(x = 190,y =295)

tk.Label(tkVent, text="Situación de BRA:", bg="#FFFFFF").place(x=10,y=325)
comboBRA = ttk.Combobox(tkVent, state="readonly")
comboBRA["values"]=[0,1,2,3]
comboBRA.place(x = 190,y =325)

tk.Label(tkVent, text="Ciudad residencia:", bg="#FFFFFF").place(x=340,y=145)
comboCity = ttk.Combobox(tkVent, state="readonly")
comboCity["values"]=["1 (Tuluá)","2 (Otro)"]
comboCity.place(x = 680,y =145)

tk.Label(tkVent, text="Tipo de zona:", bg="#FFFFFF").place(x=340,y=175)
comboTipoZona = ttk.Combobox(tkVent, state="readonly")
comboTipoZona["values"]=["1 (Urbana)","2 (Rural)"]
comboTipoZona.place(x = 680,y =175)

tk.Label(tkVent, text="Número de asignaturas matriculadas por periodo academico:", bg="#FFFFFF").place(x=340,y=205)
matrixPer = tk.Entry(tkVent,  width=6)
matrixPer.place(x = 680,y =205)

tk.Label(tkVent, text="Promedio general del estudiante:", bg="#FFFFFF").place(x=340,y=235)
promGen = tk.Entry(tkVent,  width=6)
promGen.place(x = 680,y =235)

tk.Label(tkVent, text="Promedio de creditos matriculados por periodo:", bg="#FFFFFF").place(x=340,y=265)
promCredxPer = tk.Entry(tkVent,  width=6)
promCredxPer.place(x = 680,y =265)

tk.Label(tkVent, text="Proporcion creditos matriculado por aprobados:", bg="#FFFFFF").place(x=340,y=295)
credAproba = tk.Entry(tkVent,  width=6)
credAproba.place(x = 680,y =295)

tk.Label(tkVent, text="Condición de excepcion:", bg="#FFFFFF").place(x=340,y=325)
comboExcepcion = ttk.Combobox(tkVent, state="readonly")
comboExcepcion["values"]=["0 (No tiene excepcion)","1 (Si tiene excepcion)"]
comboExcepcion.place(x = 680,y =325)

tk.Label(tkVent, text="Carlos Daniel Marín M.", bg="#FFFFFF").place(x=5,y=500)
tk.Label(tkVent, text="Andres Mosquera A.", bg="#FFFFFF").place(x=5,y=520)

boton = tk.Button(tkVent,text="Predicción", command=botonAcc, height=2, width=15)
boton.place(x=365,y=400)

tkVent.mainloop()
