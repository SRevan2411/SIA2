import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb, ListedColormap

def linear(z, derivate = False):
    a = z
    if derivate:
        da = np.ones(z.shape)
        return a,da
    return a

def logistic(z, derivate = False):
    a = 1/(1 + np.exp(-z))
    if derivate:
        da = a * (1-a)
        return a, da
    return a

class OLN:
    #one layer network initialization
    def __init__(self, n_inputs, n_outputs, activation_function):
        self.w = -1 + 2 * np.random.rand(n_outputs,n_inputs) #matriz de pesos sinapticos
        self.b = -1 + 2 * np.random.rand(n_outputs,1) #vector columna de los bias
        self.f = activation_function 
    
    def predict(self, X):
        #X es una matriz de n dimensiones x p ejemplos
        Z = np.dot(self.w, X) + self.b
        return self.f(Z)
    
    def fit(self,X,Y,epochs = 200,lr=0.1):
        ''' 
        Se utilizará broadcasting para poder hacer la actualización de los pesos
        el bias se va a concatenar n veces para poder operar esa ecuacion
        de esa manera nos ahorramos tener que propagar cada elemento uno por uno
        '''
        #lr learning rate
        p = X.shape[1] #Numero de ejemplos
        for _ in range(epochs):
            Z = np.dot(self.w, X) + self.b
            #Yest: y estimado, dY: derivada de Y
            Yest, dY = self.f(Z,derivate = True)
            #Calcular gradiente local
            lg = (Y - Yest) * dY
            #Actualizar matriz de pesos
            self.w += (lr/p) * np.dot(lg,X.T)
            self.b += (lr/p) * np.sum(lg, axis = 1).reshape(-1, 1) #chingadera que nos evita hacerlo iterativo

def MLP_binary_draw(X,n_classes,Y,net):
    hue_values = np.linspace(0, 1, n_classes, endpoint=False)
    colors = [hsv_to_rgb([hue, 1, 1]) for hue in hue_values]
    cmap_custom = ListedColormap(colors)
    for i in range(n_outputs):
    # Obtener índices de muestras que pertenecen a la clase i
        indices = np.where(Y[i] == 1)[0]
        plt.scatter(X[0, indices], X[1, indices],edgecolors='k',c=colors[i], marker='o', s=100, label='Datos de entrenamiento')


    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    # Aplanar la malla para poder pasarla al modelo
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = net.predict(grid)
    Z = np.argmax(Z, axis=0)  # Obtener la clase más probable
    # Reshape para que coincida con la malla
    Z = Z.reshape(xx.shape)

    # Dibujar las fronteras de decisión usando el mismo cmap
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_outputs + 1) - 0.5, cmap=cmap_custom)
    
    plt.show()

if __name__ == "__main__":
    entradas = pd.read_csv("Entradas.csv")
    salidas = pd.read_csv("Salidas.csv")
    X = entradas.T.values
    Y = salidas.T.values
    # Definir parámetros
    n_inputs = X.shape[0]  # 2
    n_outputs = Y.shape[0]  # 4
    learning_rate = 0.1
    print(n_outputs)
    # Inicializar la red neuronal
    red = OLN(n_inputs, n_outputs, activation_function=logistic)
    red.fit(X, Y, epochs=5000, lr=learning_rate)
    # Realizar predicciones
    MLP_binary_draw(X,n_outputs,Y,red)

    

   
    