""" utilityfunctions.py
    Archivo que contiene los metodos para knn.py

    Author: Gabriel Aldahir Lopez Soto
    Email: gabriel.lopez@gmail.com
    Institution: Universidad de Monterrey
    First created: Wed 13 May, 2020
"""
# Importa las librerias estandard y la libreria utilityfunctions
import numpy as np

import pandas as pd

from random import randint

# Se inicializa el promedio y la desviacion 
mean = []
std = []

def scale_features(dataX,label,*arg):
    """
    Aplicar el escalamiento de caracteristicas dependiendo 
    cada caracteristica

    INPUTS
    :parametro 1: matriz dataX con las caracteristicas
    :parametro 2: string con la etiqueta de que proceso se hara
    :parametro 2: argumento opcional para la media y desviacion estandar

    OUTPUTS
    :return: matriz con las caracteristicas escaladas

    """
    #Se aplica el escalamiento de caracteristicas si son los datos de entrenamiento
    if label == "training":
        meanT = dataX.mean()
        stdT = dataX.std()
        dataScaled = ((dataX - meanT ) / stdT)
        return dataScaled, meanT, stdT
    #Se aplica el escalamiento de caracteristicas si son los datos de prueba
    if label == "testing":
        dataScaled = ((dataX - arg[0] ) / arg[1])
        return dataScaled

def load_data(path_and_filename):
    """
    Cargar los archivos CSV de datos de entrenamiento, 
    desplegar los valores de entrenamiento escalados

    INPUTS
    :parametro 1: direccion y nombre del archivo

    OUTPUTS
    :return: matriz con los valores de x escalados, datosY, entrenamiento x, 
    entrenamiento y y asi como una copia de los datos de pruebas y
    """
    try:
        training_data = pd.read_csv(path_and_filename)

    except IOError:
      print ("Error: El archivo no existe")
      exit(0)

    #Se obtienen las filas y columnas
    filas = len(training_data)
    columnas = len(list(training_data))

    #Se obtiene las caracteristicas
    dataX = pd.DataFrame.to_numpy(training_data.iloc[:,0:columnas-1])

    #Se obtiene los resultados
    dataY = pd.DataFrame.to_numpy(training_data.iloc[:,-1]).reshape(filas,1)

    testingDataX = []
    testingDataY = []

    #Se obtiene el 20% del de los datos
    testingPercent = round(len(dataX)*.05)

    """
    El for lo que hace es recorre el 20% de los datos, selecciona uno de manera random, 
    ese dato se agrega a una nueva lista de datos de prueba, y se elimina de la lista original
    """
    for x in range(0,testingPercent):

        delNumber = randint(0,len(dataX)-1)

        testingDataX.append(dataX[delNumber])
        dataX = np.delete(dataX, delNumber, 0)

        testingDataY.append(dataY[delNumber])
        dataY = np.delete(dataY, delNumber, 0)

    testingDataX = np.array(testingDataX)
    testingDataY = np.array(testingDataY)

    #Se escalan los datos de entrenamiento
    dataXscaled=[]

    # Ciclo para escalar los datos de entrenamiento
    for featureX in dataX.T:
        dataScaled, meanX, stdX = scale_features(featureX,"training")
        dataXscaled.append(dataScaled)
        mean.append(meanX)
        std.append(stdX)

    dataXscaled = np.array(dataXscaled).T

    #Se escalan los datos de prueba
    dataXscaledTesting=[]

    testX = testingDataX

    # Ciclo para escalar los datos de prueba
    for featureX,meanX,stdX in zip (testingDataX.T,mean,std):
        dataScaled= scale_features(featureX,"testing",meanX,stdX)
        dataXscaledTesting.append(dataScaled)

    dataXscaledTesting = np.array(dataXscaledTesting).T

    return dataXscaled, dataY, dataXscaledTesting, testingDataY, testX

def getPredictValue(x0,x_train,y_train,k):
    """
    Obtener el valor predecido, se obtiene la distancia, despues se saca
    los k cercanos, y probabilidad de tener o no diabetes

    INPUTS
    :parametro 1: valor x0
    :parametro 2: datos de entrenamiento x
    :parametro 3: datos de entrenamiento y
    :parametro 4: valor k

    OUTPUTS
    :return: matriz con los valores de x escalados, datosY
    promedio, desviacion, columnas
    """
    # Se inicializa los valores de ceros, unos y el diccionario
    ceros = unos = 0
    distancia = {}

    # Ciclo para obtener la distancia euclidiana de cada dato
    for x in range(len(x_train)):
        distancia[x] = distanciaEuclidiana(x_train[x],x0)

    # Se ordenan los resultados
    distancia = sorted(distancia.items(), key=lambda x: x[1])

    # Ciclo para sacar los k numeros cercanos
    for x in range(k):
        data = distancia[x][0]
        
        """
        Se obtiene la posicion del numero k cercano 
        y se obtiene su resultado en los datos de entrenamiento
        """
        if(y_train[data]==0):
            ceros +=1
        else:
            unos +=1

    # Se compara cual de los valor es mayor si el cero o uno
    if ceros > unos :
        return 0, ceros/k, unos/k
    else:
        return 1, ceros/k, unos/k

def distanciaEuclidiana(x_train,x0):
    """
    Calcular la distancia Eucladiana

    INPUTS
    :parametro 1: matriz de prueba x
    :parametro 2: arreglo con los valores de x0

    OUTPUTS
    :return: distancia eucladiana
    """
    return (np.sqrt(np.sum(np.subtract(x_train,x0)**2)))

def confusionMatrix(x_testing,y_testing,x_train,y_train,k,testX):
    """
    Calcular la matriz de confusion asi como sus metricas de rendimiento

    INPUTS
    :parametro 1: matriz de prueba x
    :parametro 2: matriz de prueba y
    :parametro 3: matriz de entrenamiento x
    :parametro 4: matriz de entrenamiento y
    :parametro 5: valor entero de k

    OUTPUTS
    :return: Nada
    """
    #Declaracion de variables de matriz de confusion y metricas
    tp = tn = fn = fp = accuracy = precision = recall = specifity = f1 = 0

    #Print de los headers de resultados
    print("{:12s}".format("Pregnancies"),"{:12s}".format("Glucose"),"{:16s}".format("BloodPressure")
        ,"{:16s}".format("Skin Thickness"),"{:10s}".format("Insulin"),"{:8s}".format("BMI")
        ,"{:18s}".format("Diabetes.Ped.Fun."),"{:5s}".format("Age"),"{:14s}".format("Pb. Diabetes")
        ,"{:14s}".format("Pb. NO Diabetes"))
    
    # Variable entera para recorrer todos los numeros
    n = 0

    # Ciclo para recorrer todos los valores de prueba
    for x,y in zip(x_testing,y_testing):

        # Se calcula el valor de la prediccion, asi como porcentaje de diabetico o no
        prediccion, diabetesY, diabetesN = getPredictValue(x,x_train,y_train,k)

        # Print de los resultados en cada posicion
        print("{:12s}".format(str(round(testX[n][0],3))),"{:12s}".format(str(round(testX[n][1],3))),
            "{:16s}".format(str(round(testX[n][2],3))),"{:16s}".format(str(round(testX[n][3],3))),
            "{:10s}".format(str(round(testX[n][4],3))),"{:8s}".format(str(round(testX[n][5],3))),
            "{:18s}".format(str(round(testX[n][6],3))),"{:5s}".format(str(round(testX[n][7],3))),
            "{:>10s}".format(str(round(diabetesY,2))),"{:>10s}".format(str(round(diabetesN,2))))

        # Se aumenta +1 la posicion 
        n += 1

        #Se compara para saber si son TP, TN, FN, FP
        if (prediccion == 1 and y == 1):
            tp += 1
        if (prediccion == 0 and y == 0):
            tn += 1
        if (prediccion == 0 and y == 1):
            fn += 1 
        if (prediccion == 1 and y == 0):
            fp += 1


    #Se calculan la metricas dependiendo de los resultados de la matriz de confusion
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = (tp)/(tp + fp)
    recall = (tp/(tp + fn))
    specifity = (tn/(tn + fp))
    f1 = (2.0 * ((precision * recall)/(precision + recall)))

    print("-"*28)
    print("K ",k)
    print("-"*28)
    print("Confusion Matrix")
    print("-"*28)
    print("TP ",tp," | FP ",fp)
    print("-"*28)
    print("FN ",fn," | TN  ",tn)
    print("-"*28)

    print("Performance Metrics")
    print("-"*28)
    print("Accuracy:\t", accuracy)
    print("Precision:\t", precision)
    print("Recall:\t\t", recall)
    print("Specifity:\t", specifity)
    print("F1:\t\t", f1)

    return None
