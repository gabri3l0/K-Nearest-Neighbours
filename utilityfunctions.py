""" utilityfunctions.py
    Archivo que contiene los metodos para logistic-classification.py

    Author: Gabriel Aldahir Lopez Soto
    Email: gabriel.lopez@gmail.com
    Institution: Universidad de Monterrey
    First created: Sat 18 April, 2020
"""
# Importa las librerias estandard y la libreria utilityfunctions
import numpy as np

import pandas as pd

from random import randint

# Se inicializa el promedio y la desviacion 
mean = []

std = []

def eval_hypothesis_function(w, x):
    """
    Evaluar la funcion de hipotesis con la w y x

    INPUTS
    :parametro 1: matriz w con los paramametros
    :parametro 2: matriz x con las caracteristicas

    OUTPUTS
    :return: matriz con la funcion evaluada

    """
    return 1/(1+np.exp(np.matmul(-w.T, x)))

def remap(x):
    """
    Crea un arreglo con 1's para despues agregarlos a la matriz x
    y despues aplicarle trasnpuesta

    INPUTS
    :parametro 1: matriz x con las caracteristicas ya escaladas

    OUTPUTS
    :return: matriz con 1's agregados y transpuesta

    """
    Nr = x.shape[0]

    #Se hace el arreglo de unos y se concatena a la matriz
    x = np.hstack((np.ones((Nr,1)),x)).T
    return x

def compute_gradient_of_cost_function(x, y, w):
    """
    Calcular la funcion de costo de la gradiente descendiente
    se evaluea la funcion, despues de resta la funcion hipotesis 
    con las y, se multiplican por las x.T, se suman y se dividen 
    entre el numero de muestras, despues de hace un reshape

    INPUTS
    :parametro 1: matriz w con los parametros
    :parametro 2: matriz x con las caracteristicas
    :parametro 2: matriz y con los valores de costo ultimo milla

    OUTPUTS
    :return: matriz con el gradiente de la funcion de costo

    """
    #Se obtiene el numero de filas y de caracteristicas
    features, Nr = x.shape
    
    #Se evalua la funcion hipotesis por medio de la funcion logistica
    hypothesis_function = eval_hypothesis_function(w, x)

    #Se resta la funcion hipotesis con las valores resultado
    residual =  np.subtract(hypothesis_function.T, y)

    #Se multiplica la traspuesta del residup por las x de entrenamiento
    multi = (residual.T*x)

    #Se suma todas las x
    suma = np.sum(multi,axis=1)

    #Se divide la suma entre el numero de datos
    gradient_of_cost_function = (suma/Nr)

    #Se cambia el tamano del resultado de la division por el una matriz de 9x1
    gradient_of_cost_function = np.reshape(gradient_of_cost_function,(features,1))

    return gradient_of_cost_function


def compute_L2_norm(gradient_of_cost_function):
    """
    Calcular la L2 norm

    INPUTS
    :parametro 1: matriz de gradiente de la funcion de costo

    OUTPUTS
    :return: calcular L2 norm

    """
    return np.linalg.norm(gradient_of_cost_function)

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

def predict_log(x_testing,w):
    """
    Calcular la el valor de prediccion sobre los datos de prueba

    INPUTS
    :parametro 1: matriz x_testing con los datos de entrenamientos
    :parametro 2: matriz w con los parametros optimos

    OUTPUTS
    :return: matriz con las y predecidas

    """
    return np.matmul(w.T, x_testing)

def confusionMatrix(predicted_class,y_testing):
    """
    Calcular la matriz de confusion asi como sus metricas de rendimiento

    INPUTS
    :parametro 1: matriz de datos predecidos con los datos de entrenamientos
    :parametro 2: matriz y con los datos verdaderos

    OUTPUTS
    :return: matriz con las y predecidas

    """
    #Declaracion de variables de matriz de confusion y metricas
    tp = tn = fn = fp = accuracy = precision = recall = specifity = f1 = 0

    #Se recorren los resultados predecidos y los resultados correctos
    for x,y in zip(predicted_class,y_testing):
        #Se compara para saber si son TP, TN, FN, FP
        if (x > 0 and y == 1):
            tp += 1
        if (x < 0 and y == 0):
            tn += 1
        if (x < 0 and y == 1):
            fn += 1
        if (x > 0 and y == 0):
            fp += 1

    #Se calculan la metricas dependiendo de los resultados de la matriz de confusion
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = (tp)/(tp + fp)
    recall = (tp/(tp + fn))
    specifity = (tn/(tn + fp))
    f1 = (2.0 * ((precision * recall)/(precision + recall)))

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

def load_data(path_and_filename):
    """
    Cargar los archivos CSV de datos de entrenamiento, 
    desplegar los valores de entrenamiento escalados y hacerles 
    el remap

    INPUTS
    :parametro 1: direccion y nombre del archivo

    OUTPUTS
    :return: matriz con los valores de x escalados, datosY
    promedio, desviacion, columnas

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
    testingPercent = round(len(dataX)*.20)

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

    for featureX in dataX.T:
        dataScaled, meanX, stdX = scale_features(featureX,"training")
        dataXscaled.append(dataScaled)
        mean.append(meanX)
        std.append(stdX)

    dataXscaled = np.array(dataXscaled).T

    #Se escalan los datos de prueba
    dataXscaledTesting=[]

    for featureX,meanX,stdX in zip (testingDataX.T,mean,std):
        dataScaled= scale_features(featureX,"testing",meanX,stdX)
        dataXscaledTesting.append(dataScaled)

    dataXscaledTesting = np.array(dataXscaledTesting).T

    #Funcion que manda a llamar el llenado de unos para los datos de prueba y entrenaminto
    dataXscaledTesting = remap(dataXscaledTesting) 
    dataXscaled = remap(dataXscaled)

    #Traspuesta de las matrices
    dataXscaled = dataXscaled.T
    dataXscaledTesting = dataXscaledTesting.T

    return dataXscaled.T, dataY, dataXscaledTesting.T, testingDataY, columnas

def show_w(w):
    """
    Desplegar los valores optimos de la w

    INPUTS
    :parametro 1: matriz con los valores optimos de la w

    OUTPUTS

    """
    print("-"*28)
    print("W parameter")
    print("-"*28)
    for x in range(len(w)):
        print("w",x,":",float(w[x]))


def gradient_descent(x_training, y_training, w, stopping_criteria, learning_rate):
    """
    Calcula la gradiente descendiente y comprueba si es la optima

    INPUTS
    :parametro 1: matriz x con datos de entrenamiento
    :parametro 2: matriz y con datos de entrenamiento
    :parametro 3: matriz w con parametros optimos
    :parametro 4: criterio de paro
    :parametro 5: learning rate

    OUTPUTS
    :return: matriz w con los parametros optimos

    """
    L2_norm = 100.0
    
    #Ciclo para parar el programa
    while L2_norm > stopping_criteria:

        #Funcion para caluclar el costo del gradiente 
        gradient_of_cost_function = compute_gradient_of_cost_function(x_training,y_training,w)

        #Se obtienen las w
        w = w - learning_rate*gradient_of_cost_function

        #Se ajusta la norma l2 para para el criterio de paro
        L2_norm = compute_L2_norm(gradient_of_cost_function)

    return w