#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trabajo 1
Álvaro Fernández García.
Grupo 3.
"""

import numpy as np
import matplotlib.pyplot as plt


def simula_unif(N=2, dims=2, size=(0, 1)):
    m = np.random.uniform(low=size[0], 
                          high=size[1], 
                          size=(N, dims))
    
    return m


def label_data(x1, x2):
    y = np.sign((x1-0.2)**2 + x2**2 - 0.6)
    idx = np.random.choice(range(y.shape[0]), size=(int(y.shape[0]*0.1)), replace=True)
    y[idx] *= -1
    
    return y 


def coef2line(w):
    if(len(w)!= 3):
        raise ValueError('Solo se aceptan rectas para el plano 2d. Formato: [<a0>, <a1>, <b>].')
    
    a = -w[0]/w[1]
    b = -w[2]/w[1]
    
    return a, b


def plot_data(X, y, w):
    #Preparar datos
    a, b = coef2line(w)
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = grid.dot(w)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$w^tx$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white', label='Datos')
    ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Solucion')
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel='Intensidad promedio', ylabel='Simetria')
    ax.legend()
    plt.title('Solucion ejercicio 2.1')
    plt.show()
 
 
 #Coloca los pesos como es necesario para la funcion anterior:
def permutateWeight(w):
	 w2 = w.copy()
	 aux = np.float64(0)
	 aux = w2[0]
	 w2[0] = w2[1]
	 w2[1] = w2[2]
	 w2[2] = aux
	
	 return w2
 
 
# Fijar la semilla:
np.random.seed(8)


########################################################################################################################
# 1:
# Ejercicio sobre la búsqueda iterativa de óptimos
########################################################################################################################


# EJERCICIO 1.1


# Implementación del Gradiente Descendente: 
def GD1(X, w, n, gd_func, MAX_ITERS):
	index = -1 #Si no supera el umbral
	for i in range(MAX_ITERS):
		w = w - n * gd_func(X, w)
		if(E(w[0], w[1]) < 10**(-14)):
			index = i
			break
		
	return index, w[0], w[1]


#--------------------------------------------------------------------------


# EJERCICIO 1.2:


#Definición de la función E(u,v):
def E(u,v):
	return (((u**3) * np.exp(v-2)) - (4 * (v**3) * np.exp(-u))) **2


# Cálculo del gradiente para la función del ejercicio 1.2 b):
def gd_E(X, w):
	u = w[0]
	v = w[1]
	# Como esta parte es igual para ambas, la calculamos una sola vez:
	aux = 2 * ((u**3) * np.exp(v-2) -  4 * (v**3) * np.exp(-u))
	du = aux * (3 * np.exp(v-2) * (u**2) + 4 * (v**3) * np.exp(-u))
	dv = aux * ((u**3) * np.exp(v-2) - 12 * (v**2) * np.exp(-u))
	
	return np.array([du, dv], np.float64)


"""
Apartados b) y c):
nu = 0.05
Maximo de iteraciones = 10000
"""
print("Parte 1: Ejercicio 2: Apartados b) y c)")
X = np.array([0,0], np.float64)
w = np.array([1,1], np.float64)
i,u,v = GD1(X, w, 0.05, gd_E, 10000)
print("Iteraciones: {}\nu: {}\nv: {}\nE(u,v): {}".format(i,u,v, E(u,v)))

z = input()
#--------------------------------------------------------------------------


# EJERCICIO 1.3


# Implementación de f(x,y)
def f(x,y):
	return (x-2)**2 + 2*(y+2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)


# Cálculo del gradiente para la función del ejercicio 1.3:
def gd_f(X, w):
	x = w[0]
	y = w[1]
	dx = 2*x - 4 + (4 * np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y))
	dy = 4*y + 8 + (4 * np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y))
	
	return np.array([dx, dy], np.float64)


# Implementación del GD pero con las gráficas y los mínimos incorporados:
def GD2(X, w, n, gd_func, MAX_ITERS):
	out = []
	minval = 1000000
	minx = 0
	miny = 0
	#Añadir el valor inicial:
	out.append(f(w[0], w[1]))
	for i in range(MAX_ITERS):
		w = w - n * gd_func(X, w)
		out.append(f(w[0], w[1]))
		
		if(f(w[0], w[1]) < minval):
			minval = f(w[0], w[1])
			minx = w[0]
			miny = w[1]
		
	return out, minx, miny 


"""
Apartado a):
	Numero de iteraciones: 50
	1ª ejecucion: nu = 0.01
	2ª ejecucion: nu = 0.1 
"""
print("Parte 1: Ejercicio 3: Apartado a)")
X = np.array([0,0], np.float64)
w = np.array([1,1], np.float64)
out1,a,b = GD2(X, w, 0.01, gd_f, 50)


w = np.array([1,1], np.float64)
out2,a,b = GD2(X, w, 0.1, gd_f, 50)


# Dibujar la gráfica:
X_AXIS = range(0, 51)
plt.plot(X_AXIS, out1, label='n=0.01')
plt.plot(X_AXIS, out2, label='n=0.1')
plt.xlabel("Iteraciones")
plt.ylabel("f")
plt.legend()
plt.show()


"""
Apartado b):
	Como no se mencionan que valores de nu e iteraciones utilizar
	realizare las ejecuciones con los ultimos valores enunciados:
		nu = 0.1
		Numero iteraciones = 50
"""
print("Parte 1: Ejercicio 3: Apartado b)")
n = 0.1
MAX_ITERS = 50

w = np.array([2.1,-2.1], np.float64)
tmp1,x1,y1 = GD2(X, w, n, gd_f, MAX_ITERS)

w =np.array([3,-3], np.float64)
tmp2,x2,y2 = GD2(X, w, n, gd_f, MAX_ITERS)

w = np.array([1.5,1.5], np.float64)
tmp3,x3,y3 = GD2(X, w, n, gd_f, MAX_ITERS)

w = np.array([1,-1], np.float64)
tmp4,x4,y4 = GD2(X, w, n, gd_f, MAX_ITERS)


# Generar la tabla:
table1 = "| x0=2.1 | y0=-2.1 | x={} | y={} | f={} |".format(x1, y1, f(x1, y1))
table2 = "| x0=  3 | y0=  -3 | x={} | y={} | f={} |".format(x2, y2, f(x2, y2))
table3 = "| x0=1.5 | y0= 1.5 | x={} | y={} | f={} |".format(x3, y3, f(x3, y3))
table4 = "| x0=  1 | y0=  -1 | x={} | y={} | f={} |".format(x4, y4, f(x4, y4))

# Imprimir:
print(table1)
print(table2)
print(table3)
print(table4)

z = input()

########################################################################################################################
# 2:
# Ejercicio sobre Regresión Lineal
########################################################################################################################

# EJERCICIO 2.1:

# Implementacion del algoritmo de la pseudoinversa:
# w = Pseduo-Inv(X) * y
def PseudoInverse(X, y):
	# Calcular la pseudoinversa de X:
	pseudoIvn = np.linalg.pinv(X)
	# Clacular w:
	w = pseudoIvn.dot(y)
	return w 


# Error en el conjunto de aprendizaje:
# Simplemente calcula la formula del Ein para unos datos de
# entrada y unos pesos: 1/N * Sum(wt*Xn - yn)**2
def Ein(X, y, w):
	out = np.float64(0)
	for n in range(X.shape[0]):
		out += ((w.transpose()).dot(X[n]) - y[n])**2	
	
	return (1/X.shape[0]) * out
	

# Error en las predicciones del test:
# Calcula el error en los datos predichos con la formula:
# 1/N * Sum(ypn - yn)**2
def Eout(Yorig, Ypredd):
	out = np.float64(0)
	for n in range(X.shape[0]):
		out += (Ypredd[n] - Yorig[n])**2	
	
	return (1/Yorig.shape[0]) * out


# Funcion que dado un array de predicciones reales lo etiqueta:
def tag(Y):
	Ytag = []
	for i in range(Y.shape[0]):
		if(Y[i] < 0):
			Ytag.append(-1)
		else:
			Ytag.append(1)

	return np.array(Ytag, np.float64)


# Funcion para predecir:
# Contiene un array con todos los elementos a predecir:
# Implementacion de la funcion h(x) = wt*x
# Aplica esa funcion a todos los datos de X y devuelve un 
# array con las predicciones
def predict(X, w):
	y = [] # Predicciones
	for n in range(X.shape[0]):
		y.append((w.transpose()).dot(X[n]))
	
	return np.array(y, np.float64) 


# Calcula el error de clasificacion:
# Devuelve los errores en las predicciones de la clase -1 y los de la 1
def ErrClass(X, w, y):
	Ytemp = predict(X, w)
	Y = tag(Ytemp) #Predicciones:
	err_1 = 0
	err1 = 0
	for i in range(Y.shape[0]):
		if (Y[i] == 1 and y[i] == -1):
			err_1+=1
		elif (Y[i] == -1 and y[i] == 1):
			err1+=1

	return err_1, err1

	
# Funcion de perdida:
# Es el gradiente, se trata de la derivada de Ein:
# 2/N * Sum(Xnj * (wt*xn -yn))
def lossF(X, y, w):
	out = []
	
	pre_grad = X.dot(w) - y
	
	for j in range(w.shape[0]):
		val = (X[:,j] * pre_grad).sum()
		val *= 2/float(X.shape[0])
		out.append(val)
			
	return np.array(out, np.float64)
	

# Implementacion del gradiente desdendente estocastico:
def SGD(X, y, w, n, gd_func, MAX_ITERS, BATCHSIZE):
	for _ in range(MAX_ITERS):
		# Barajar la muestra
		idx = np.arange(X.shape[0])
		np.random.shuffle(idx)
		X = X[idx]
		y = y[idx]
		# Iterar en los batches
		for i in range(0, X.shape[0], BATCHSIZE):
			w = w - n * gd_func(X[i:i+BATCHSIZE], y[i:i+BATCHSIZE], w)
	
	return w


# Cargar los datos:
X = np.load("data/X_train.npy")
y = np.load("data/y_train.npy")

# Elegir aquellos que utilizaremos, solo 1 y 5:
X = X[(y==1) + (y==5)]
y = y[(y==1) + (y==5)]
# Sustituir la etiqueta 5, por -1
y[y==5] = -1
#Añadir una columna de 1 al principio de X para el termino independiente:
Xorig = X.copy() #Guardamos la matriz sin los unos, ya que sera necesaria para el plot
X = np.c_[np.ones(X.shape[0]), X]


print("Parte 2: Ejercicio 1:")


"""
Entrenar los algoritmos:
	Para el gradiente descendente estocastico se ha elegido:
	learning rate = 0.01
	Numero iteraciones = 100
	Tamaño de los batches = 128
"""
w_pi = PseudoInverse(X,y)
w_sgd = SGD(X, y, np.array([0,0,0], np.float64), 0.01, lossF, 100, 128)

print("w segun la pseudoinversa: {}".format(w_pi))
print("w segun el gradiente descendente estocastico: {}".format(w_sgd))

#Calcular el error del ajuste:
print("RESUMEN TRAINING:")
print("Ein para el ajuste de la pseudoinversa: {}".format(Ein(X,y,w_pi)))
print("Ein para el ajuste del gradiente: {}".format(Ein(X,y,w_sgd)))


#Calcular el error de clases:
err_pi = ErrClass(X, w_pi, y)
err_sgd = ErrClass(X, w_sgd, y)
erroresTotales_pi = err_pi[0] + err_pi[1]
erroresTotales_sgd = err_sgd[0] + err_sgd[1]

print("Error de clases para el ajuste de la pseudoinversa:") 
print("\tPredicciones de -1 erroneas {}".format(err_pi[0]))  
print("\tPredicciones de 1 erroneas {}".format(err_pi[1]))
print("\tTotal de errores: {} de {} muestras".format(erroresTotales_pi, X.shape[0]))

print("Error de clases para el ajuste del gradiente:")
print("\tPredicciones de -1 erroneas {}".format(err_sgd[0]))  
print("\tPredicciones de 1 erroneas {}".format(err_sgd[1]))
print("\tTotal de errores: {} de {} muestras".format(erroresTotales_sgd, X.shape[0]))

#Dibujar las graficas de Training
waux = permutateWeight(w_pi)
plot_data(Xorig, y, waux)
waux = permutateWeight(w_sgd)
plot_data(Xorig, y, waux)


# Cargar los datos para el test:
X = np.load("data/X_test.npy")
y = np.load("data/y_test.npy")

# Elegir aquellos que utilizaremos:
X = X[(y==1) + (y==5)]
y = y[(y==1) + (y==5)]
# Sustituir la etiqueta 5, por -1
y[y==5] = -1
#Añadir una columna de 1 al principio de X para el termino independiente:
Xorig = X.copy()
X = np.c_[np.ones(X.shape[0]), X]

# Predecir con los pesos de la pseudoinversa:
yp_pi = predict(X, w_pi)
# Predecir con los pesos del gradiente:
yp_sgd = predict(X, w_sgd)

# Etiquetar los resultados predichos:
yptag_pi = tag(yp_pi)
yptag_sgd = tag(yp_sgd)


# Calcular el Eout:
print("RESUMEN TEST:")
print("Eout para el ajuste de la pseudoinversa: {}".format(Eout(y,yp_pi)))
print("Eout para el ajuste del gradiente: {}".format(Eout(y,yp_sgd)))

#Calcular el error de clases:
err_pi = ErrClass(X, w_pi, y)
err_sgd = ErrClass(X, w_sgd, y)
erroresTotales_pi = err_pi[0] + err_pi[1]
erroresTotales_sgd = err_sgd[0] + err_sgd[1]

print("Error de clases para el ajuste de la pseudoinversa:") 
print("\tPredicciones de -1 erroneas {}".format(err_pi[0]))  
print("\tPredicciones de 1 erroneas {}".format(err_pi[1]))
print("\tTotal de errores: {} de {} muestras".format(erroresTotales_pi, X.shape[0]))

print("Error de clases para el ajuste del gradiente:")
print("\tPredicciones de -1 erroneas {}".format(err_sgd[0]))  
print("\tPredicciones de 1 erroneas {}".format(err_sgd[1]))
print("\tTotal de errores: {} de {} muestras".format(erroresTotales_sgd, X.shape[0]))

#Dibujar las graficas de Test:
waux = permutateWeight(w_pi)
plot_data(Xorig, y, waux)
waux = permutateWeight(w_sgd)
plot_data(Xorig, y, waux)

z = input()
#--------------------------------------------------------------------------


# EJERCICIO 2.2:

print("Parte 2: Ejercicio 2")

# Apartados a) y b):
X = simula_unif(N=1000, dims=2, size=(-1, 1))
y = label_data(X[:, 0], X[:, 1])

# Dibujar los datos:
plt.scatter(X[:, 0], X[:, 1])
plt.title("Mapa de puntos 2D")
plt.xlabel("Primera caracteristica")
plt.ylabel("Segunda caracteristica")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Mapa de etiquetas")
plt.xlabel("Primera caracteristica")
plt.ylabel("Segunda caracteristica")
plt.show()

# Apartado c)
#Añadir a la muestra la columna de unos inicial:
X = np.c_[np.ones(X.shape[0]), X]

"""
Ajuste utilizando SGD
	learning rate = 0.01
	Numero iteraciones = 100
	Tamaño de los batches = 128
"""
w = SGD(X, y, np.array([0,0,0], np.float64), 0.01, lossF, 100, 128)
print("Estimacion de los pesos: {}".format(w))
print("Ein obtenido: {}".format(Ein(X, y, w)))

z = input()


"""
 Apartado d)
"""
# Declarar variables:
MeanEin = []     #Guardara los Ein
MeanEout = []    #Eout
errneg1In = []   #Numero de predicciones de -1 erroneas
errneg1Out = []
err1In = []	     #Numero de predicciones de 1 erroneas
err1Out = []


print("Parte 2: Ejercicio 2, apartado d)")
print("Iniciando 1000 experimentos...")
for _ in range(1000):
	#Generar muestra:
	X = simula_unif(N=1000, dims=2, size=(-1, 1))
	y = label_data(X[:, 0], X[:, 1])
	#Añadir a la muestra la columna de unos inicial:
	X = np.c_[np.ones(X.shape[0]), X]
	#Realizar el ajuste:
	# nu = 0.01, 30 iteraciones, 128 de batchsize
	w = SGD(X, y, np.array([0,0,0], np.float64), 0.01, lossF, 30, 128)
	#Calcular los Errores:
	MeanEin.append(Ein(X, y, w))
	err = ErrClass(X,w,y)
	errneg1In.append(err[0])
	err1In.append(err[1])
	#Generar muestra para test:
	X = simula_unif(N=1000, dims=2, size=(-1, 1))
	y = label_data(X[:, 0], X[:, 1])
	X = np.c_[np.ones(X.shape[0]), X]
	#Predecir:
	yp = predict(X, w)
	#Calcular los errores:
	MeanEout.append(Eout(y, yp))
	err = ErrClass(X,w,y)
	errneg1Out.append(err[0])
	err1Out.append(err[1])
	
	
#Calcular la media:
aux = np.array(MeanEin, np.float64)
mean = aux.mean()
print("Ein medio: {}".format(mean))

aux = np.array(MeanEout, np.float64)
mean = aux.mean()
print("Eout medio: {}".format(mean))

aux = np.array(errneg1In, np.float64)
mean1 = aux.mean()
print("Media de predicciones erroneas de -1 en el train: {}".format(mean1))

aux = np.array(err1In, np.float64)
mean2 = aux.mean()
print("Media de predicciones erroneas de 1 en el train: {}".format(mean2))	
	
aux = mean1 + mean2
print("Media de errores totales en el train: {}".format(aux))

aux = np.array(errneg1Out, np.float64)
mean1 = aux.mean()
print("Media de predicciones erroneas de -1 en el test: {}".format(mean1))

aux = np.array(err1Out, np.float64)
mean2 = aux.mean()
print("Media de predicciones erroneas de 1 en el test: {}".format(mean2))	
	
aux = mean1 + mean2
print("Media de errores totales en el test: {}".format(aux))