#!/usr/bin/python3.7
""" knn.py
    Algoritmo que implementa los vecinos k mas cercanos

    Author: Gabriel Aldahir Lopez Soto
    Email: gabriel.lopez@gmail.com
    Institution: Universidad de Monterrey
    First created: Wed 13 May, 2020
"""

def main():
	"""
	Aqui se manda llamar el archivo para leer los datos del CSV, asi mismo
	se obtienen las x, y de entrenamiento, el promedio de las x aplicando 
	escalamiento de caracteristicas y la desviacion estandar para despues
	obtener la matriz de confusion para poder predecir

	Datos de entrada:
	Nada

	Datos de salida:
	Nada
	"""
	# Importa las librerias estandard y la libreria utilityfunctions
	import numpy as np
	
	import utilityfunctions as uf

	# Metodo para obtener el x,y de entrenamiento, x, y de pruebas, copia de X
	x_train, y_train, x_testing, y_testing, testX = uf.load_data('diabetes.csv')

	# Numero de la k
	k = 10

	# Metodo para obtener la matriz de confusion
	uf.confusionMatrix(x_testing,y_testing,x_train,y_train,k,testX)

main()