#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""""""""""""""""""""""""""""""""""""""""
Trabajo 3
Álvaro Fernández García.
Grupo 3.
"""""""""""""""""""""""""""""""""""""""""

import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
REGRESIÓN
* Pruebas aerodinámicas y acústicas de dos y tres 
secciones de álabes aerodinámicos realizadas en un túnel 
de viento.
* Dados los datos de las pruebas, determina el nivel 
de presión de sonido escalado, en decibelios.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("REGRESIÓN")

#Constantes:
PARTICION = 0.2         #Para hacer la partición entre train y test
SEED = 88               #Semilla de números aleatorios
SPLITS = 5              #Divisiones en kFold

#Fijar la semilla:
np.random.seed(SEED)

#Cargar los datos:
X = np.load("datos/airfoil_self_noise_X.npy")
y = np.load("datos/airfoil_self_noise_y.npy")

#Imprimir información sobre los datos:
N = X.shape[0]
print("[!] Información sobre el dataset:")
print("\t[*] Tamaño de la muestra: " + str(N))
print("\t[*] Número de características: " + str(X.shape[1]))

"""
########################################################################
Probar con distintos polinomios:
    * Modelo : Ridge (Necesario elegir un Alpha)
########################################################################
"""

#Guardar los datos originales:
X_original = X.copy()
alphas = []
scores = []
poly = []

for n in range(2,8):
    print("\n[+] Grado de características polinómicas = " + str(n))

    #Preprocesado:
    X = PolynomialFeatures(degree=n).fit_transform(X_original)
    X = StandardScaler().fit_transform(X)

    #Partir los datos:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=PARTICION, random_state=SEED)

    kfold = model_selection.KFold(n_splits=SPLITS)
    
    print("\t[+] Eligiendo alpha y el coeficiente del polinomio para Ridge:")

    # Se eligen órdenes de magnitud para los posibles valores de alpha:
    alphaValues = [10**-4, 10**-3, 10**-2, 10**-1, 10**1, 10**2, 10**3, 10**4]

    bestScore = -10**6

    # Validación cruzada para cada alpha:
    for alpha in alphaValues:
        model = linear_model.Ridge(alpha=alpha)
        crossVres = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        meanScore = crossVres.mean()
        print("\t\t[*] Alpha = " + str(alpha) + " score: " + str(meanScore))
        if(meanScore > bestScore):
            bestScore = meanScore
            bestAlpha = alpha

    print("\t[+] Mejor alpha encontrado para grado = " + str(n) + ": " + str(bestAlpha) + " con un coeficiente R^2 de %.5f" % bestScore)

    scores.append(bestScore)
    alphas.append(bestAlpha)
    poly.append(n)
    

#Mejor alpha y mejor n encontrado:
index = np.argmax(np.array([scores], np.float64))
finalAlpha = alphas[index]
finalN = poly[index]
print("\n[!] Mejor alpha encontrado = " + str(finalAlpha) + ". Mejor grado de polinomio = " + str(finalN))

"""
########################################################################
Una vez elegido el alpha y el n, entrenamos el modelo con todos los
datos:
########################################################################
"""

#Preprocesado:
X = PolynomialFeatures(degree=finalN).fit_transform(X_original)
X = StandardScaler().fit_transform(X)

#Partir los datos:
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=PARTICION, random_state=SEED)

print("\n[!] Se tomarán " + str(PARTICION * 100) + "% de datos para train y " + str((1-PARTICION) * 100) + "% para test")
print("\t[*] Tamaño del conjunto de entrenamiento: " + str(X_train.shape[0]))
print("\t[*] Tamaño del conjunto de test: " + str(X_test.shape[0]))

RIDGE = linear_model.Ridge(alpha=finalAlpha)

RIDGE.fit(X_train, y_train)

print("\n\t[*] Coeficiente R^2 para train: %.5f" % RIDGE.score(X_train, y_train))

"""
########################################################################
Realizar el test:
########################################################################
"""

print("\n[!] Realizando test...")
print("\t[*] Coeficiente R^2 para test: %.5f" % RIDGE.score(X_test, y_test))

input("Pulse cualquier tecla para continuar...")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CLASIFICACIÓN
* Problema de clasificación de dígitos escritos a mano
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("\nCLASIFICACIÓN")

#Constantes:
PENALTY = 'l2'          #Mejores alpha y penalización encontrados, para no repetir el cálculo
ALPHA = 10              #de nuevo, ya que consume bastante tiempo.

#Reestablecer la semilla:
np.random.seed(SEED)

#Cargar los datos de train
X_train = np.load("datos/optdigits_tra_X.npy")
y_train = np.load("datos/optdigits_tra_y.npy")

N = X_train.shape[0]

print("[!] Información sobre el dataset:")
print("\t[*] Tamaño de la muestra: " + str(N))
print("\t[*] Número de características: " + str(X_train.shape[1]))

"""
#########################################################################
Preprocesado de los datos
#########################################################################
"""

# Las características son enteros, por lo que es necesario un casting:
X_train = StandardScaler().fit_transform(np.float64(X_train[:,:]))

"""
#########################################################################
Como el número de características es elevado, 
puede ser interesante aplicar regularización L1 para determinar 
las características relevantes.
También probaremos regularización L2 y nos quedaremos con la mejor de
las dos.
    * Ambas penalizaciones se aplican a Regresión Logística
#########################################################################
"""

opt = input("[!] ¿Realizar la búsqueda de los hiperparámetros? [s/n]: ")

if(opt == 's'):
    print("[+] Buscando mejor penalización (l1, o l2) y alpha...")

    # Se eligen órdenes de magnitud para los posibles valores de alpha:
    alphaValues = np.array([10**-4, 10**-3, 10**-2, 10**-1, 10**1, 10**2, 10**3, 10**4], np.float64)

    # Se eligen las penalizaciones:
    penalty = ['l1', 'l2']

    hyperparam = dict(C=alphaValues, penalty=penalty)

    #Divisiones para validación cruzada:
    kfold = model_selection.KFold(n_splits=SPLITS)

    #Modelo que utilizaremos:
    LR = linear_model.LogisticRegression()

    #Buscar la mejor combinación
    clf = GridSearchCV(LR, hyperparam, cv=kfold, scoring='accuracy', n_jobs=-1)
    model = clf.fit(X_train, y_train)

    #Extraer resultados: 
    bestPenalization = model.best_estimator_.get_params()['penalty']
    bestAlpha = model.best_estimator_.get_params()['C']
    
else:
    bestPenalization = PENALTY
    bestAlpha = ALPHA

print("\t[*] Mejor penalización encontrada: " + str(bestPenalization) + " con Alpha = " + str(bestAlpha))

print("[+] Entrenando modelo...")

finalLR = linear_model.LogisticRegression(penalty=bestPenalization, C=bestAlpha)
finalLR.fit(X_train, y_train)

print("\t[*] Porcentaje de aciertos en Train = %.4f%%" % (finalLR.score(X_train, y_train) * 100))

# Matriz de confusión:
y_pred = finalLR.predict(X_train)
print("\n[+] Matriz de confusión en Train:\n")
print(confusion_matrix(y_train, y_pred))

"""
#########################################################################
Probar el modelo con los datos de test:
#########################################################################
"""

print("\n[!] Realizando test...")

X_test = np.load("datos/optdigits_tes_X.npy")
y_test = np.load("datos/optdigits_tes_y.npy")

X_test = StandardScaler().fit_transform(np.float64(X_test[:,:]))

print("\t[*] " + str(X_test.shape[0]) + " datos en test")

# Calcular el Error de clasificación en el test:
finalScore = finalLR.score(X_test, y_test)

print("\t[*] Porcentaje de aciertos en Test = %.4f%%" % (finalScore * 100))

# Matriz de confusión:
y_pred = finalLR.predict(X_test)
print("\n[+] Matriz de confusión en Test:\n")
print(confusion_matrix(y_test, y_pred))