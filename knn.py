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
	x_train, y_train, x_testing, y_testing = uf.load_data('diabetes.csv')

	# print(x_train)
	# print(x_testing[0])
	k = 3
	distance = uf.getDistance(x_testing[0],x_train,k)
	# print(len(distance))
	print(distance)
	exit(1)

	# Metodo para imprimir las w optimas
	uf.show_w(w)

	# Metodo para obtener las clases pedecidas
	predicted_class = uf.predict_log(x_testing,w)

	# Transponer los resultados para compararlos
	predicted_class = predicted_class.T

	# Metodo obtener la matriz de confusion y sus metricas de rendimiento
	uf.confusionMatrix(predicted_class,y_testing)

main()