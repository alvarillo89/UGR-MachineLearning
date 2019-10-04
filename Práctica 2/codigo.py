#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trabajo 2
Álvaro Fernández García.
Grupo 3.
"""

import numpy as np
import matplotlib.pyplot as plt
from Funciones_necesarias_p2 import *


#Fijar la semilla
np.random.seed(8)  


# Funciones auxiliares para la practica:

# Implementacion de la funcion simula uniforme,
def simula_unif(N=2, dims=2, size=(0, 1)):
    m = np.random.uniform(low=size[0], 
                          high=size[1], 
                          size=(N, dims))
    
    return m


########################################################################################################################
# 1:
# Ejercicio sobre la complejidad de H y el ruido
########################################################################################################################

# Funcion para etiquetar:
def tag(X, a, b):
	# Calcular la funcion para todos los x de X
	y = np.sign(X[:, 1] - a * X[:, 0] - b)
				
	return y
		

# Para cambiar aleatoriamente el 10% de las etiquetas:
def modify_label(x1, x2, y):
    idx = np.random.choice(range(y.shape[0]), size=(int(np.rint(y.shape[0]*0.1))), replace=True)
    y[idx] *= -1
    
    return y
	

# Función proporcionada para pintar datos y la frontera de decisión de un hiperplano:
def plot_datos_cuad(X, y, fz, fx1, title='Point clod plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white', label='Datos')
	 #Evitar warning de raices negativas
    #ax.plot(grid[:, 0], fx1(grid[:, 0]), 'black', linewidth=2.0, label='Solucion')
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    ax.legend()
    plt.title(title)
    plt.show()


# Definición de funciones necesarias para el ejercicio 1.3
   
def fa(X):
	return (X[:,0] - 10)**2 + (X[:,1] - 20)**2 - 400

def fax0(x0):
	return 20 + np.sqrt(400 - (x0 - 10)**2)

def fb(X):
	return 0.5 * (X[:,0] + 10)**2  + (X[:,1] - 20)**2 - 400

def fbx0(x0):
	return 20 + np.sqrt(400 - 0.5 * (x0 + 10)**2)

def fc(X):
	return 0.5 * (X[:,0] - 10)**2 - (X[:,1] + 20)**2 - 400

def fcx0(x0):
	return np.sqrt(0.5 * (x0 - 10)**2 - 400) - 20

def fd(X):
	return X[:,1] - 20 * X[:,0]**2 - 5 * X[:,0] + 3

def fdx0(x0):
	return 20 * x0**2 + 5 * x0 - 3


# EJERCICIO 1.1

print("Parte 1, Ejercicio 1, apartados a y b:")

Xu = simula_unif(N=50, dims=2, size=(-50, 50))
plt.scatter(Xu[:, 0], Xu[:, 1])
plt.title("Nube de puntos para Simula Uniforme")
plt.show()

Xg = simula_gaus(size=(50,2), sigma=(5,7))
plt.scatter(Xg[:, 0], Xg[:, 1])
plt.title("Nube de puntos para Simula Gauss")
plt.show()

z = input("Pulse cualquier tecla...")


# EJERCICIO 1.2

print("Parte 1, Ejercicio 2, Apartado a")
a,b = simula_recta()
y2a = tag(Xu, a, b)
plot_datos_recta(Xu, y2a, a, b, title="Simula unif etiquetados")

print("Parte 1, Ejercicio 2, Apartado b")
# Añadir los cambios aleatorios a la muestra:
#Cambiar 10% de las positivas:
Xpos = Xu[y2a==1]
ypos = modify_label(Xpos[:,0], Xpos[:,1], y2a[y2a==1])
#Cambiar 10% de las negativas:
Xneg = Xu[y2a==-1]
yneg = modify_label(Xneg[:,0], Xneg[:,1], y2a[y2a==-1]) 

XuMod = np.concatenate((Xpos, Xneg), axis=0)
y2b = np.concatenate((ypos, yneg), axis=0)
plot_datos_recta(XuMod, y2b, a, b, title="Simula unif modificados")

z = input("Pulse cualquier tecla...")


# EJERCICIO 1.3

print("Parte 1, Ejercicio 3")
plot_datos_cuad(XuMod, y2b, fa, fax0, "1.3, Apartado a)")
plot_datos_cuad(XuMod, y2b, fb, fbx0, "1.3, Apartado b)")
plot_datos_cuad(XuMod, y2b, fc, fcx0, "1.3, Apartado c)")
plot_datos_cuad(XuMod, y2b, fd, fdx0, "1.3, Apartado d)")

z = input("Pulse cualquier tecla...")


########################################################################################################################
# 2:
# Modelos lineales
########################################################################################################################

# Implementacion el algoritmo del perceptron:
def ajusta_PLA(X, y, MAX_ITERS, w0):
	w = w0
	iters = MAX_ITERS # Por si no lo encontrara
	for n in range(1, MAX_ITERS+1):
		waux = w.copy()
		
		for i in range(X.shape[0]):
			if(np.sign((w.transpose()).dot(X[i])) != y[i]):
				w = w + y[i] * X[i]
		
		if(np.all(np.abs(w-waux)<10**(-6))):
			iters = n
			break
			
	return w, iters


# Implementacion del sigmoide
def sigmoid(x):
	return 1 / float(1 + np.exp(-x))


# Implementacion del gradiente de la regresion lineal logistica:
def grad(X, y, w):
	aux = np.zeros(w.shape[0], np.float64)
	
	for i in range(X.shape[0]):
		aux += -y[i] * X[i] * sigmoid(-y[i] * (w.transpose()).dot(X[i]))
		
	aux *= 1/float(X.shape[0])
	
	return aux

	
# Error para la regresion lineal logistica:
def Err(X, y, w):
	add = np.float64(0)
	for i in range(X.shape[0]):
		add += np.log(1 + np.exp(-y[i] * (w.transpose()).dot(X[i])))
		
	add *= 1/float(X.shape[0])
	
	return add 

	
# Implementacion del gradiente desdendente estocastico:
def SGD(X, y, w, n, gd_func, MAX_ITERS, BATCHSIZE):
	# Las epocas:
	for _ in range(MAX_ITERS):
		waux = w.copy()
		# Barajar la muestra
		idx = np.arange(X.shape[0])
		np.random.shuffle(idx)
		X = X[idx]
		y = y[idx]
		# Iterar en los batches
		for i in range(0, X.shape[0], BATCHSIZE):
			w = w - n * gd_func(X[i:i+BATCHSIZE], y[i:i+BATCHSIZE], w)
	
		if(np.all(np.abs(w-waux) < 0.01)):
			break

	return w


# Funcion para predecir:
def predict(X, w):
	ytmp = X.dot(w)
	y = []
	for i in range(ytmp.shape[0]):
		if(sigmoid(ytmp[i]) < 0.5):
			y.append(0)
		else:
			y.append(1)
	
	return y
	
	
# Error de clases:
def ErrClass(X, w, y):
	ypredict = predict(X, w)
	err = 0
	for i in range(y.shape[0]):
		if(y[i] != ypredict[i]):
			err+=1
	
	return err
		

# EJERCICIO 2.1
# Apartado a)
print("Parte 2, Ejercicio 1, apartado a)")
print("NOTA: Numero maximo de iteraciones = 1000")
# Añadir los unos:
Xu1 = np.c_[np.ones(Xu.shape[0]), Xu]
# Apartado a) Con ceros
w, i = ajusta_PLA(Xu1, y2a, 1000, np.array([0,0,0], np.float64))
print("Se necesita un total de {} iteraciones con w = (0,0,0)".format(i))
# Apartado b) con muestras aleatorias:
meani = []
for _ in range(10):
	w = np.random.uniform(0, 1, 3)
	w, i = ajusta_PLA(Xu1, y2a, 1000, w)
	meani.append(i)

meaninp = np.array(meani, np.float64)
i = meaninp.mean()
print("Se necesita un total de {} iteraciones con w aleatorio".format(i))


# Apartado b)
print("Parte 2, Ejercicio 1, apartado b)")
print("NOTA: Numero maximo de iteraciones = 1000")
XuMod1 = np.c_[np.ones(XuMod.shape[0]), XuMod]
# Apartado a) Con ceros
w, i = ajusta_PLA(XuMod1, y2b, 1000, np.array([0,0,0], np.float64))
print("Se necesita un total de {} iteraciones con w = (0,0,0)".format(i))
# Apartado b) con muestras aleatorias:
meani = []
for _ in range(10):
	w = np.random.uniform(0, 1, 3)
	w, i = ajusta_PLA(XuMod1, y2b, 1000, w)
	meani.append(i)

meaninp = np.array(meani, np.float64)
i = meaninp.mean()
print("Se necesita un total de {} iteraciones con w aleatorio".format(i))

z = input("Pulse cualquier tecla...")

print("Parte 2, Ejercicio 2)")
		 
#Crear datos para train:
X_train = simula_unif(N=100, dims=2, size=(0,2))
X1 = np.c_[np.ones(X_train.shape[0]), X_train]
a,b = simula_recta(intervalo=(0,2))
y_train = tag(X_train, a, b)
plot_datos_recta(X_train, y_train, a, b, "Datos Generados para Train")

"""
	Ajustar: SGD, tasa aprendizaje = 0.01, 1000 epocas como mucho
				Tamaño del batch 32
"""
w = SGD(X1, y_train, np.zeros(3, np.float64), 0.01, grad, 1000, 32)
print("Ein para los datos: {}".format(Err(X1, y_train, w)))
#Cambiar las etiquetas -1 por 0
y_train[y_train==-1] = 0
print("Error de clasificacion para los datos de train: {}".format(ErrClass(X1, w, y_train)))

#Crear datos para test:
X_test = simula_unif(N=2000, dims=2, size=(0,2))
X1 = np.c_[np.ones(X_test.shape[0]), X_test]
y_test = tag(X_test, a, b)
plot_datos_recta(X_test, y_test, a, b, "Datos Generados para Test")
print("Eout para los datos: {}".format(Err(X1, y_test, w)))
#Cambiar las etiquetas -1 por 0
y_test[y_test==-1] = 0
print("Error de clasificacion para los datos de test: {}".format(ErrClass(X1, w, y_test)))

z = input("Pulse cualquier tecla...")

########################################################################################################################
# Bonus
########################################################################################################################

def coef2line(w):
    if(len(w)!= 3):
        raise ValueError('Solo se aceptan rectas para el plano 2d. Formato: [<a0>, <a1>, <b>].')
    
    a = -w[0]/w[1]
    b = -w[2]/w[1]
    
    return a, b
 
 #Coloca los pesos como es necesario para la funcion anterior:
def permutateWeight(w):
	 w2 = w.copy()
	 aux = np.float64(0)
	 aux = w2[0]
	 w2[0] = w2[1]
	 w2[1] = w2[2]
	 w2[2] = aux
	
	 return w2

# Implementacion del algoritmo de la pseudoinversa:
# w = Pseduo-Inv(X) * y
def PseudoInverse(X, y):
	# Calcular la pseudoinversa de X:
	pseudoIvn = np.linalg.pinv(X)
	# Clacular w:
	w = pseudoIvn.dot(y)
	return w 


#Error de clases:
def ErrClass(X, y, w):
	err = 0
	pred = np.sign(X.dot(w)) 
	for n in range(X.shape[0]):
		if(pred[n] != y[n]):
			err+=1
	
	return err


# Simplemente calcula la formula del Ein para unos datos de
# entrada y unos pesos: 1/N * Sum(wt*Xn - yn)**2
def Err(X, y, w):

	out = ((X.dot(w) - y)**2).sum()
	out *= float(1/X.shape[0]) 
	
	return out


# Implementacion el algoritmo del perceptron pocket:
def PLA_pocket(X, y, MAX_ITERS, w0, Error):
	w = w0
	best = w0
	for n in range(1, MAX_ITERS+1):
		waux = w.copy()
		
		for i in range(X.shape[0]):
			if(np.sign((w.transpose()).dot(X[i])) != y[i]):
				w = w + y[i] * X[i]
		
		if(Error(X, y, w) < Error(X, y, best)):
			best = w.copy() 
		
		if(np.all(np.abs(w-waux)<10**(-6))):
			break
			
	return best


print("Bonus")

# Preparar el conjunto de train:
X = np.load("datos/X_train.npy")
y = np.load("datos/y_train.npy")

# Elegir aquellos que utilizaremos, solo 4 y 8:
X = X[(y==4) + (y==8)]
y = y[(y==4) + (y==8)]
# Sustituir la etiqueta 4, por -1
y[y==4] = -1
# Sustituir la etiqueta 8, por 1
y[y==8] = 1
#Añadir una columna de 1 al principio de X para el termino independiente:
Xorig = X.copy() #Guardamos la matriz sin los unos, ya que sera necesaria para el plot
X1 = np.c_[np.ones(X.shape[0]), X]

#Como modelo lineal utilizaremos la pseudo-inversa:
w = PseudoInverse(X1, y)

#Ahora lo mejoramos con el perceptron pocket:
w = PLA_pocket(X1, y, 2000, w, ErrClass)

#Dibujar:
w2 = permutateWeight(w)
a,b = coef2line(w2)
plot_datos_recta(X, y, a, b, "Datos para train: PLA_POCKET")
print("Ein: {}".format(Err(X1,y,w)))
print("Error de clasificacion para train: {} de {} ({})".format(ErrClass(X1,y,w), y.shape[0], ErrClass(X1,y,w) / y.shape[0]))

# Preparar el conjunto de test:
X = np.load("datos/X_test.npy")
y = np.load("datos/y_test.npy")

# Elegir aquellos que utilizaremos, solo 4 y 8:
X = X[(y==4) + (y==8)]
y = y[(y==4) + (y==8)]
# Sustituir la etiqueta 4, por -1
y[y==4] = -1
# Sustituir la etiqueta 8, por 1
y[y==8] = 1
#Añadir una columna de 1 al principio de X para el termino independiente:
Xorig = X.copy() #Guardamos la matriz sin los unos, ya que sera necesaria para el plot
X1 = np.c_[np.ones(X.shape[0]), X]

plot_datos_recta(X, y, a, b, "Datos para test: PLA_POCKET")
print("Etest: {}".format(Err(X1,y,w)))
print("Error de clasificacion para test: {} de {} ({})".format(ErrClass(X1,y,w), y.shape[0], ErrClass(X1,y,w) / y.shape[0]))