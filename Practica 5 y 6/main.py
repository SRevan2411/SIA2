import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb, ListedColormap
from threading import *
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn import metrics
from customtkinter import filedialog
from activations import *

#Definicion de la red y sus funciones de activación
FilePath1 = None
FilePath2 = None
figura = plt.figure()
ax = figura.add_subplot(111)


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
    
    def fit(self, X,Y, epocas = 1000, learning_rate = 0.1):
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
            self.grafica(X,Y)

    
    def grafica(self,X,Y):
        ax.clear()
   
    
        x_min, y_min = np.min(X[0,:])-0.5,np.min(X[1,:])-0.5
        x_max, y_max = np.max(X[0,:])+0.5,np.max(X[1,:])+0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        # Aplanar la malla para poder pasarla al modelo
        grid = [xx.ravel(), yy.ravel()]
        zz = self.predict(grid)
        # Reshape para que coincida con la malla
        zz = zz.reshape(xx.shape)
        # Dibujar las fronteras de decisión usando el mismo cmap
        ax.contourf(xx, yy, zz,alpha=0.8,cmap = plt.cm.RdBu)

        """ for i in range(X.shape[1]):
            if Y[0,i]==0:
                ax.scatter(X[0,i],X[1,i],edgecolors='k',c='red', marker='o', s=100)
            else:
                ax.scatter(X[0,i],X[1,i],edgecolors='k',c='blue', marker='o', s=100) """
        # Filtra los índices para cada clase
        class_0 = Y[0] == 0
        class_1 = Y[0] == 1

        # Grafica todos los puntos de la clase 0
        ax.scatter(X[0, class_0], X[1, class_0], edgecolors='k', c='red', marker='o', s=100)

        # Grafica todos los puntos de la clase 1
        ax.scatter(X[0, class_1], X[1, class_1], edgecolors='k', c='blue', marker='o', s=100)
        canvas.draw() 




def getFilepath1():
    global FilePath1
    FilePath1 = filedialog.askopenfile()
    if FilePath1 != None:
        print(FilePath1.name)

def getFilepath2():
    global FilePath2
    FilePath2 = filedialog.askopenfile()
    if FilePath2 != None:
        print(FilePath2.name)

def generarPlano(frame,n_classes):
    global ax, canvas, figura
    ax.clear()
    if FilePath2 != None and FilePath1 != None:
        for i in range(X.shape[1]):
            if Y[0,i]==0:
                ax.scatter(X[0,i],X[1,i],edgecolors='k',c='red', marker='o', s=100)
            else:
                ax.scatter(X[0,i],X[1,i],edgecolors='k',c='blue', marker='o', s=100)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True)

def PuntosExtras():
    matrizVentana = ctk.CTkToplevel(ventana)
    matrizVentana.title("Matriz de confusion")
    frameTitle = ctk.CTkFrame(matrizVentana)
    tituloEmergente = ctk.CTkLabel(frameTitle,text="MATRIZ DE CONFUSION",font=("Arial",25),padx=10)
    tituloEmergente.grid(row=0,column=1)
    frameTitle.pack(side="top",fill=ctk.BOTH,expand=True)


    Y_pred = red.predict(X)

    # Para clasificación multi-etiqueta, establecer un umbral (por ejemplo, 0.5)
    Y_pred_binary = np.argmax(Y_pred,axis=0)
    Y_binary = np.argmax(Y,axis=0)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    """ fig,axes = plt.subplots(1, n_outputs, figsize=(10,5))
    for i in range(n_outputs):
        # Crear la matriz de confusión para cada columna
        matrizConfusion = metrics.confusion_matrix(Y[i, :], Y_pred_binary[i, :])

        mostrarmatriz = metrics.ConfusionMatrixDisplay(confusion_matrix = matrizConfusion)
        mostrarmatriz.plot(cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Etiqueta {i+1}') """
    
    confusionMatrix = metrics.confusion_matrix(Y_binary,Y_pred_binary)
    mostrarmatriz = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionMatrix)
    mostrarmatriz.plot(cmap='Blues', ax=axes)
    #frame para insertar la grafica
    frameMatriz = ctk.CTkFrame(matrizVentana)
    frameMatriz.pack(side="left",fill=ctk.BOTH,expand=True)
    #Crear la figura
    #Insertar figura en tkinter
    Ncanvas = FigureCanvasTkAgg(fig, master=frameMatriz)
    Ncanvas.draw()
    Ncanvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True)

                         


def Proceso():
    global red
    entradas = pd.read_csv(FilePath1.name)
    salidas = pd.read_csv(FilePath2.name)
    X = entradas.T.values
    Y = salidas.T.values
    # Definir parámetros
    red = multicapaDensa((2,300,1))
    red.fit(X, Y)

    

def Threading():
    t1=Thread(target=Proceso)
    t1.start()

def showPlots():
    global X
    global Y
    entradas = pd.read_csv(FilePath1.name)
    salidas = pd.read_csv(FilePath2.name)
    X = entradas.T.values
    Y = salidas.T.values
    # Definir parámetros
    n_inputs = X.shape[0]  # 2
    n_outputs = Y.shape[0]  # 4
    learning_rate = 0.1
    generarPlano(frame,n_outputs)

    
#INTERFAZ GRAFICA
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
ventana = ctk.CTk()    
#Titulo
frameTitle = ctk.CTkFrame(ventana)
Title = ctk.CTkLabel(frameTitle,text="RED MULTICAPA",padx=300,font=("Arial",25),pady=10)
Title.grid(row=0,column=1)
frameTitle.pack(side="top",fill=ctk.BOTH,expand=True)

#INTERFAZ ENTRADAS
frameInputs = ctk.CTkFrame(ventana)
#PARAMETRO DE APRENDIZAJE
#boton
btn= ctk.CTkButton(frameInputs,text = 'TRAIN', command = Threading)
btn.grid(row=7,column=1,pady=20,padx=20)


btnf1= ctk.CTkButton(frameInputs,text = 'Inputs File', command = getFilepath1)
btnf1.grid(row=9,column=1,pady=20,padx=20) 

btnf2= ctk.CTkButton(frameInputs,text = 'Outputs File', command = getFilepath2)
btnf2.grid(row=10,column=1,pady=20,padx=20)

btnf2= ctk.CTkButton(frameInputs,text = 'ShowGraph', command = showPlots)
btnf2.grid(row=11,column=1,pady=20,padx=20)

frameInputs.pack(side='right',fill=ctk.BOTH,expand=True)

frame = ctk.CTkFrame(ventana)
frame.pack(side='left')
canvas = FigureCanvasTkAgg(figura, master=frame)

if __name__ == "__main__":
    generarPlano(frame,0)
    ventana.protocol("WM_DELETE_WINDOW", quit)
    ventana.mainloop()