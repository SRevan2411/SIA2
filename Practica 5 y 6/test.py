import numpy as np
import matplotlib.pyplot as plt
from multicapa import *
import pandas as pd

def grafica(X,Y,red):
    plt.figure()
   
    
    x_min, y_min = np.min(X[0,:])-0.5,np.min(X[1,:])-0.5
    x_max, y_max = np.max(X[0,:])+0.5,np.max(X[1,:])+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    # Aplanar la malla para poder pasarla al modelo
    grid = [xx.ravel(), yy.ravel()]
    zz = red.predict(grid)
    # Reshape para que coincida con la malla
    zz = zz.reshape(xx.shape)
    plt.xlim([x_min,x_max])
    plt.ylim([y_min,y_max])
    # Dibujar las fronteras de decisión usando el mismo cmap
    plt.contourf(xx, yy, zz,alpha=0.8,cmap = plt.cm.RdBu)

    for i in range(X.shape[1]):
        if Y[0,i]==0:
            plt.scatter(X[0,i],X[1,i],edgecolors='k',c='red', marker='o', s=100)
        else:
            plt.scatter(X[0,i],X[1,i],edgecolors='k',c='blue', marker='o', s=100)
    plt.show()


entradas = pd.read_csv("entradas.csv")
salidas = pd.read_csv("salidas.csv")
X = entradas.T.values
Y = salidas.T.values


red = multicapaDensa((2,200,1))
grafica(X,Y,red)
red.fit(X,Y)
grafica(X,Y,red)