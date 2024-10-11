import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from threading import *

w = np.array([random.random(),random.random()])
entradas = np.empty((0,1),dtype=float)
salidaDeseada = np.empty((0,1),dtype=float)

class Adaline():
    def __init__(self,w,ax):
        self.w = w
        self.ax = ax
    
    def predict(self,x,activacion):
        v = np.dot(self.w,x.T)
        input
        match activacion:
            case 'lineal' : 
                return v
            case 'sigmoide': 
                return 1/(1+np.exp(-v))
            case 'tanh':
                return np.tanh(v)
    
    def train(self,x,d,etha,activacion,epocas):
        for z in range(epocas):
            v = np.dot(self.w,x.T)
            ecm = np.mean((d-v)**2)
            if ecm < 0.45:
                break
            e = d-v

            print(f"Deseada: {d}, Actual: {v}")


            self.w[0] = self.w[0] + etha * np.sum(e)
            self.w[1] = self.w[1] + etha * np.sum(e * x[:,1])
            
            print(self.w)
          
            ax.clear()
            ax.set_xlim(-5,5)
            ax.set_ylim(-5,5)
            #Actualizar labels
            B.set(str(round(self.w[0],2)))
            W1.set(str(round(self.w[1],2)))
            #Generar ejes de coordenadas
            nv = np.dot(self.w,x.T)
            ax.axhline(0, color='black',linewidth=1)
            ax.axvline(0, color='black',linewidth=1)
            ax.plot(x[:,1],d,'Pg')
            ax.plot(x[:,1],nv)
            canvas.draw()
           
        
        ecm = np.mean((d-v)**2)
        
        


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
        ndatos = np.array([[x]])
        entradas = np.vstack((entradas,x))
        if event.button == MouseButton.LEFT:
            ax.plot(x,y,'Pg')
            salidaDeseada = np.append(salidaDeseada,y)
        canvas.draw()
        print(salidaDeseada.shape)

def Entrenar():
    etha = float(TETHA.get())
    maxepocas = int(TEPOCS.get())
    global entradas,w,salidaDeseada
    print(entradas.shape)
    entradas = np.hstack((np.ones((entradas.shape[0],1)),entradas))
    print(entradas)
    neurona = Adaline(w,ax)
    neurona.train(entradas,salidaDeseada,etha,'lineal',maxepocas)
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
Title = ctk.CTkLabel(frameTitle,text="ADALINE REGRESIÓN",padx=300,font=("Arial",25),pady=10)
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
    
    x = np.array([[1],[2],[3],[4]])
    d = np.array([-3,1,2,6])
    x = np.hstack((np.ones((x.shape[0],1)),x))
    for i in range(100):
        v = np.dot(w,x.T)
        etha = 0.03
        error = d-v
        w[0] = w[0] + etha * np.sum(error)
        w[1] = w[1] + etha * np.sum(error * x[:,1])
        
    v = np.dot(w,x.T)
    print(v)
    
    #neurona = Adaline(w)
    #neurona.plot_contour(x,'sigmoide')
    #neurona.train(x,d,0.3,'sigmoide',100)
    #neurona.plot_contour(x,'sigmoide')
    generarPlano(frame)
    ventana.protocol("WM_DELETE_WINDOW", quit)
    ventana.mainloop()

  


