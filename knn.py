#!/usr/bin/python3.7
""" multivariate-linear-regression.py
    Algoritmo que implementa clasificacion logistica

    Author: Gabriel Aldahir Lopez Soto
    Email: gabriel.lopez@gmail.com
    Institution: Universidad de Monterrey
    First created: Sat 18 April, 2020
"""

def main():
	"""
	Aqui se manda llamar el archivo para leer los datos del CSV, asi mismo
	se obtienen las x, y de entrenamiento, el promedio de las x aplicando 
	escalamiento de caracteristicas y la desviacion estandar para despues
	obtener los parametros w y con base eso usar los datos de prueba y
	predecir las y o el costo de la ultima milla

	Datos de entrada:
	Nada

	Datos de salida:
	Nada
	"""
	# Importa las librerias estandard y la libreria utilityfunctions
	import numpy as np
	
	import utilityfunctions as uf

	# Metodo para obtener el x,y de entrenamiento, promedio, desviacion estandar, y caracteristicas
	x_train, y_train, x_testing, y_testing, testX = uf.load_data('diabetes.csv')

	# print(x_train)
	# print(x_testing[0])
	k = 3
	# prediccion = uf.getPredictValue(x_testing[0],x_train,y_train,k)

	uf.confusionMatrix(x_testing,y_testing,x_train,y_train,k,testX)

main()