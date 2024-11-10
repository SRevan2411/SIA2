import numpy as np
from activations import *

class multicapaDensa:
    def __init__(self,arquitectura,activacion_oculta=tanh,activacion_salida=logistic):
        #Obtener cuantas capas tiene la arquitectura
        self.L = len(arquitectura)-1
        #Lista con las matrices de los pesos y el bias
        self.w = [None] * (self.L+1)
        self.b = [None] * (self.L+1)
        #Lista con las funciones de activacion
        self.f = [None] * (self.L+1)

        #Inicializar pesos
        for l in range(1,self.L+1):
            self.w[l] = -1 + 2 * np.random.rand(arquitectura[l],arquitectura[l-1])
            self.b[l] = -1 + 2 * np.random.rand(arquitectura[l],1)
            
            #asignar funciones de activacion
            if l == self.L:
                self.f[l] = activacion_salida
            else:
                self.f[l] = activacion_oculta
    
    def predict(self,X):
        a = np.asanyarray(X)
        for l in range(1,self.L + 1):
            z = np.dot(self.w[l],a) + self.b[l]
            a = self.f[l](z)
        return a
    
    def fit(self, X,Y, epocas = 500, learning_rate = 0.1):
        #numero de ejemplos
        P = X.shape[1]

        for _ in range(epocas):
            for p in range(P):
                #inicializar activaciones
                a = [None] * (self.L + 1)
                da = [None] * (self.L + 1)
                #gradiente local
                lg = [None] * (self.L + 1)

                #Propagacion
                a[0] = X[:,p].reshape(-1,1)
                for l in range(1,self.L+1):
                    z = np.dot(self.w[l], a[l-1])+self.b[l]
                    a[l],da[l] = self.f[l](z,derivate = True)
                
                #retropropagación
                for l in range(self.L,0,-1):
                    if l == self.L:
                        lg[l]=(Y[:,p].reshape(-1,1)-a[l])*da[l]
                    else:
                        lg[l]=np.dot(self.w[l+1].T,lg[l+1])*da[l]
                
                #gradiente descendente (ajustar pesos y bias)
                for l in range(1,self.L+1):
                    self.w[l] += learning_rate * np.dot(lg[l],a[l-1].T)
                    self.b[l] += learning_rate * lg[l]






