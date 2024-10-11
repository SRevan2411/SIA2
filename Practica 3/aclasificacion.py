import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from threading import *

w = np.array([random.random(),random.random(),random.random()])
entradas = np.empty((0,2),dtype=float)
salidaDeseada = np.empty((0,1),dtype=float)

class Adaline():
    def __init__(self,w,ax):
        self.w = w
        self.ax = ax
    
    def predict(self,x,activacion):
        v = np.dot(self.w,x.T)
        match activacion:
            case 'lineal' : 
                return v
            case 'sigmoide': 
                return 1/(1+np.exp(-v))
            case 'tanh':
                return np.tanh(v)
    
    def train(self,x,d,etha,activacion,epocas):
        self.plot_contour(d,x,activacion)
        for z in range(epocas):
            v = self.predict(x,activacion)
            
            ecm = np.mean((d-v)**2)
            if ecm < 0.03:
                break
            match activacion:
                case 'lineal':
                    dv = 1
                case 'sigmoide':
                    dv = v * (1 - v)
                case 'tanh':
                    dv = 1 - np.power(v, 2)

        
            e = d-v 

            

            self.w[0] = self.w[0] + etha * np.sum(e*dv)
            self.w[1] = self.w[1] + etha * np.sum(e *dv* x[:,1])
            self.w[2] = self.w[2] + etha * np.sum(e *dv* x[:,2])
            '''
            for i in range (self.w.shape[0]):
                self.w[i] = self.w[i] + etha * np.sum(e * dv * x[:, i])
                print(self.w[i])

            '''
            B.set(str(round(self.w[0],2)))
            W1.set(str(round(self.w[1],2)))
            W2.set(str(round(self.w[2],2)))
            self.plot_contour(d,x,activacion)

            
    def plot_contour(self,d,x,activacion):
        # Generar una cuadrícula de valores de entrada
        x1_range = np.linspace(-5, 5, 200)
        x2_range = np.linspace(-5, 5, 200)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        # Aplanar la cuadrícula para que coincida con las entradas del modelo
        x_test = np.hstack((np.ones((X1.ravel().shape[0], 1)), X1.ravel().reshape(-1, 1), X2.ravel().reshape(-1, 1)))

        # Predecir los valores para toda la cuadrícula
        Z = self.predict(x_test, activacion).reshape(X1.shape)

        # Generar la gráfica de contorno
        self.ax.clear()
        self.ax.contourf(X1, X2, Z, levels=50, cmap='RdYlBu')
        self.ax.scatter(x[:, 1], x[:, 2],c=d, edgecolors='k', cmap='RdYlBu', marker='o', s=100, label='Datos de entrenamiento')
        self.ax.legend()
        canvas.draw()     


#FUNCIONES GRAFICOS
def generarPlano(frame):
    global ax,canvas
    figura = plt.figure(figsize=(5,5),dpi=100)
    ax = figura.add_subplot(111)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)

    #Generar ejes de coordenadas
    ax.axhline(0, color='black',linewidth=1)
    ax.axvline(0, color='black',linewidth=1)

    canvas = FigureCanvasTkAgg(figura, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True)
    #CONECTAR CON FUNCION DE EVENTO
    canvas.mpl_connect('button_press_event',onclick)

def onclick(event):
    global entradas,salidaDeseada
    x = event.xdata
    y = event.ydata

    #comprobar que no se salga del limite del plano
    if x != None and y != None:
        ndatos = np.array([[x,y]])
        entradas = np.vstack((entradas,ndatos))
        if event.button == MouseButton.LEFT:
            ax.plot(x,y,'Pg')
            salidaDeseada=np.append(salidaDeseada,0)
        elif event.button == MouseButton.RIGHT:
            ax.plot(x,y,'Pb')
            salidaDeseada=np.append(salidaDeseada,1)
        canvas.draw()
        print(salidaDeseada.shape)

def Entrenar():
    etha = float(TETHA.get())
    maxepocas = int(TEPOCS.get())
    global entradas,w,salidaDeseada
    entradas = np.hstack((np.ones((entradas.shape[0],1)),entradas))
    print(entradas)
    neurona = Adaline(w,ax)
    neurona.train(entradas,salidaDeseada,etha,'sigmoide',maxepocas)
    print("acabo")

def Threading():
    t1=Thread(target=Entrenar)
    t1.start()

#INTERFAZ GRAFICA
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
ventana = ctk.CTk()    
#Titulo
frameTitle = ctk.CTkFrame(ventana)
Title = ctk.CTkLabel(frameTitle,text="ADALINE CLASIFICACION",padx=300,font=("Arial",25),pady=10)
Title.grid(row=0,column=1)
frameTitle.pack(side="top",fill=ctk.BOTH,expand=True)


#INTERFAZ ENTRADAS
frameInputs = ctk.CTkFrame(ventana)


TITULO = ctk.CTkLabel(frameInputs,text="DEFINIR VALORES",padx=10)
TITULO.grid(row=0,column=1)

#W1
LW1 = ctk.CTkLabel(frameInputs,text="W1",padx=10)
LW1.grid(row=1,column=0)
W1 = ctk.StringVar()
W1.set(str(round(w[1],2)))
TW1 = ctk.CTkLabel(frameInputs,textvariable=W1)
TW1.grid(row=1,column=1,pady=20)

#W2
LW2 = ctk.CTkLabel(frameInputs,text="W2",padx=10)
LW2.grid(row=2,column=0)
W2 = ctk.StringVar()
W2.set(str(round(w[2],2)))
TW2 = ctk.CTkLabel(frameInputs,textvariable=W2)
TW2.grid(row=2,column=1)

#BIAS
LB = ctk.CTkLabel(frameInputs,text="BIAS",padx=10)
LB.grid(row=3,column=0)
B = ctk.StringVar()
B.set(str(round(w[0],2)))
TB = ctk.CTkLabel(frameInputs,textvariable=B)
TB.grid(row=3,column=1,pady=20,padx=20)

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

frameInputs.pack(side='right',fill=ctk.BOTH,expand=True)

frame = ctk.CTkFrame(ventana)
frame.pack(side='left')




if __name__ == "__main__":
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    d = np.array([0,1,1,1])
    x = np.hstack((np.ones((x.shape[0],1)),x))
    print(d.shape)
    for i in range(x.shape[1]):
        print(x[:,i])
    #neurona = Adaline(w)
    #neurona.plot_contour(x,'sigmoide')
    #neurona.train(x,d,0.3,'sigmoide',100)
    #neurona.plot_contour(x,'sigmoide')
    generarPlano(frame)
    ventana.protocol("WM_DELETE_WINDOW", quit)
    ventana.mainloop()

  


