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

#Definicion de la red y sus funciones de activación
FilePath1 = None
FilePath2 = None
figura = plt.figure()
ax = figura.add_subplot(111)


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
    
    def fit(self,X,Y,epochs = 200,lr=0.1,n_classes=0):
        ''' 
        Se utilizará broadcasting para poder hacer la actualización de los pesos
        el bias se va a concatenar n veces para poder operar esa ecuacion
        de esa manera nos ahorramos tener que propagar cada elemento uno por uno
        '''
        #lr learning rate
        p = X.shape[1] #Numero de ejemplos
        tol=0.07
        for _ in range(epochs):
            Z = np.dot(self.w, X) + self.b
            #Yest: y estimado, dY: derivada de Y
            Yest, dY = self.f(Z,derivate = True)
            #Calcular gradiente local
            lg = (Y - Yest) * dY
            
            #Actualizar matriz de pesos
            self.w += (lr/p) * np.dot(lg,X.T)
            self.b += (lr/p) * np.sum(lg, axis = 1).reshape(-1, 1) #chingadera que nos evita hacerlo iterativo
            self.MLP_binary_draw(X,n_classes,Y)
            mse = np.mean((Y - Yest)**2)
          
            if mse < tol:
                break
        

    #Función de graficación

    def MLP_binary_draw(self,X,n_classes,Y):
        ax.clear()
        #Definción del custom color map
        hue_values = np.linspace(0, 1, n_classes, endpoint=False)
        colors = [hsv_to_rgb([hue, 1, 1]) for hue in hue_values]
        cmap_custom = ListedColormap(colors)
        for i in range(n_classes):
        # Obtener índices de muestras que pertenecen a la clase i
            indices = np.where(Y[i] == 1)[0]
            ax.scatter(X[0, indices], X[1, indices],edgecolors='k',c=colors[i], marker='o', s=100)


        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        # Aplanar la malla para poder pasarla al modelo
        grid = np.c_[xx.ravel(), yy.ravel()].T
        Z = self.predict(grid)
        Z = np.argmax(Z, axis=0)  # Obtener la clase más probable
        # Reshape para que coincida con la malla
        Z = Z.reshape(xx.shape)

        # Dibujar las fronteras de decisión usando el mismo cmap
        ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap_custom)
    
        ax.legend()
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
        hue_values = np.linspace(0, 1, n_classes, endpoint=False)
        colors = [hsv_to_rgb([hue, 1, 1]) for hue in hue_values]
        for i in range(n_classes):
        # Obtener índices de muestras que pertenecen a la clase i
            indices = np.where(Y[i] == 1)[0]
            ax.scatter(X[0, indices], X[1, indices],edgecolors='k',c=colors[i], marker='o', s=100, label='Datos de entrenamiento')
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
    learning_rate = float(TETHA.get())
    maxepocas = int(TEPOCS.get())
    entradas = pd.read_csv(FilePath1.name)
    salidas = pd.read_csv(FilePath2.name)
    X = entradas.T.values
    Y = salidas.T.values
    # Definir parámetros
    n_inputs = X.shape[0]  # 2
    n_outputs = Y.shape[0]  # 4
    red = OLN(n_inputs, n_outputs, activation_function=logistic)
    red.fit(X, Y, epochs=maxepocas, lr=learning_rate,n_classes=n_outputs)

    

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
Title = ctk.CTkLabel(frameTitle,text="RED UNICAPA",padx=300,font=("Arial",25),pady=10)
Title.grid(row=0,column=1)
frameTitle.pack(side="top",fill=ctk.BOTH,expand=True)

#INTERFAZ ENTRADAS
frameInputs = ctk.CTkFrame(ventana)
#PARAMETRO DE APRENDIZAJE
LETHA = ctk.CTkLabel(frameInputs,text="ETHA",padx=10)
LETHA.grid(row=4,column=0)
ETHA = ctk.StringVar()
TETHA = ctk.CTkEntry(frameInputs,textvariable=ETHA)
TETHA.grid(row=4,column=1,pady=20,padx=20)

#PARAMETRO DE EPOCAS
LEPOCS = ctk.CTkLabel(frameInputs,text="EPOCAS",padx=10)
LEPOCS.grid(row=5,column=0)
EPOCS = ctk.StringVar()
TEPOCS = ctk.CTkEntry(frameInputs,textvariable=EPOCS)
TEPOCS.grid(row=5,column=1,padx=20)
#boton
btn= ctk.CTkButton(frameInputs,text = 'PLOT', command = Threading)
btn.grid(row=7,column=1,pady=20,padx=20)

btn2= ctk.CTkButton(frameInputs,text = 'Datos Adicionales', command = PuntosExtras)
btn2.grid(row=8,column=1,pady=20,padx=20)

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