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

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as skM

def botonAcc():
    #Datos
    datos = pd.read_csv('base_datos.csv', encoding = "ISO-8859-1", engine='python')
    df=pd.DataFrame(datos)
    
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
    
    #Variables predicción
    codigoPred= float(comboCodProg.get())
    
    if comboTipoProg.get() == "1 (Pregrado)":
        tipoProgPred= 1.0
    else:
        tipoProgPred= 2.0
    
    if comboJornada.get() == "1 (Diurno)":
        jornadaPred= 1.0
    else:
        jornadaPred= 2.0
        
    if comboPeriodoAdmi.get() == "1 (Primer periodo)":
        periodoAdPred= 1.0
    else:
        periodoAdPred= 2.0
    
    if comboSexo.get() == "1 (Masculino)":
        sexoPred= 1.0
    else:
        sexoPred= 2.0
      
    edadIngPred= float(edad.get())
    
    braPred= float(comboBRA.get())
    
    if comboExcepcion.get() == "0 (No tiene excepcion)":
        excepPred= 0.0
    else:
        excepPred= 1.0
        
    if comboCity.get() == "1 (Tuluá)":
        residencPred= 1.0
    else:
        residencPred= 2.0
    
    if comboTipoZona.get() == "1 (Urbana)":
        zonaPred= 1.0
    else:
        zonaPred= 2.0 
        
    matriculasxperPred= float(matrixPer.get())
    
    promedioGenPred= float(promGen.get())
    
    promedioCredPred= float(promCredxPer.get())
    
    credAprobPred= float(credAproba.get())
    
    gradoPred= 0.0 
        
    #Fin datos prediccion
    
    #Árbol de decisión
    #Se pasan los datos al modelo
    algo = DecisionTreeClassifier(criterion='entropy', splitter="best", max_depth=10)
    algo.fit(X_train, y_train)
    
    #datos ingresados sobre el estudiante
    student = np.array([codigoPred,tipoProgPred,jornadaPred,periodoAdPred,sexoPred,edadIngPred,braPred,excepPred,residencPred,zonaPred,matriculasxperPred,promedioGenPred,promedioCredPred,credAprobPred,gradoPred])
    
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
