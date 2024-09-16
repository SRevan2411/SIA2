import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from sklearn import metrics
from threading import *

#Definir estilos custom tkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


entradas = np.empty((0,2),dtype=float)
salidaDeseada = np.array([])

ea = 0
w1 = random.random()
w2 = random.random()
bias = random.random()
epocas = 0

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
        print(salidaDeseada)


    
def Threading():
    t1=Thread(target=Entrenar)
    t1.start()
    
def PuntosExtras():
    print(salidaActual)
    print(salidaDeseada)
    Tp = np.sum((salidaActual == 1) & (salidaDeseada == 1))
    Fp = np.sum((salidaActual == 1) & (salidaDeseada == 0))
    Fn = np.sum((salidaActual == 0) & (salidaDeseada == 1))
    Tn = np.sum((salidaActual == 0) & (salidaDeseada == 0))
    try:
        accuracy = (Tp+Tn)/(Tp+Tn+Fp+Fn)
        presicion = Tp/(Tp+Fp)
        sensibilidad = Tp/(Tp+Fn)
        f1_score = (2*(presicion*sensibilidad))/(presicion+sensibilidad)
    except:
        print("Uno de los datos intenta dividirse entre cero")
    print(Tp)
    print(Fp)
    print(Fn)
    print(Tn)
    matrizVentana = ctk.CTkToplevel(ventana)
    matrizVentana.title("Matriz de confusion")
    frameTitle = ctk.CTkFrame(matrizVentana)
    tituloEmergente = ctk.CTkLabel(frameTitle,text="MATRIZ DE CONFUSION",font=("Arial",25),padx=10)
    tituloEmergente.grid(row=0,column=1)
    frameTitle.pack(side="top",fill=ctk.BOTH,expand=True)

    matrizConfusion = metrics.confusion_matrix(salidaDeseada,salidaActual)
    mostrarmatriz = metrics.ConfusionMatrixDisplay(confusion_matrix = matrizConfusion, display_labels = [0, 1])
    
    #frame para insertar la grafica
    frameMatriz = ctk.CTkFrame(matrizVentana)
    frameMatriz.pack(side="left",fill=ctk.BOTH,expand=True)
    #Crear la figura
    confusionfigura = plt.figure(figsize=(5,5),dpi=100)
    nax = confusionfigura.add_subplot(111)
    mostrarmatriz.plot(ax=nax) #Renderizar en el axis nax
    
    #Insertar figura en tkinter
    Ncanvas = FigureCanvasTkAgg(confusionfigura, master=frameMatriz)
    Ncanvas.draw()
    Ncanvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True)

    #Mostrar los otros datos
    frameDatos = ctk.CTkFrame(matrizVentana)
    frameDatos.pack(side="right",fill=ctk.BOTH,expand=True)

    LPresicion = ctk.CTkLabel(frameDatos,text="Presicion: ",font=("Arial",14),padx=10)
    LPresicion.grid(row=0,column=0)
    Varpresicion = ctk.StringVar()
    Varpresicion.set(str(round(presicion,2)))
    TPres = ctk.CTkLabel(frameDatos,textvariable=Varpresicion,font=("Arial",14))
    TPres.grid(row=0,column=1,padx=10)

    Lf1 = ctk.CTkLabel(frameDatos,text="F1_Score: ",padx=10,font=("Arial",14))
    Lf1.grid(row=1,column=0)
    Varf1 = ctk.StringVar()
    Varf1.set(str(round(f1_score,2)))
    Tf1 = ctk.CTkLabel(frameDatos,textvariable=Varf1,font=("Arial",14))
    Tf1.grid(row=1,column=1,padx=10)

    LAcc = ctk.CTkLabel(frameDatos,text="Accuracy: ",padx=10,font=("Arial",14))
    LAcc.grid(row=2,column=0)
    Varacc = ctk.StringVar()
    Varacc.set(str(round(accuracy,2)))
    TAcc = ctk.CTkLabel(frameDatos,textvariable=Varacc,font=("Arial",14))
    TAcc.grid(row=2,column=1,padx=10)
                         
                         

   
    

def Entrenar():
    global entradas,w1,w2,bias,epocas,salidaActual,ea
    etha = float(TETHA.get())
    maxepocas = float(TEPOCS.get())
    print(etha)
    epocas = 0
    while epocas < maxepocas:
        ea += 1
        EA.set(str(round(ea,2)))
        m = -w1/w2
        c = -bias/w2
        x = np.linspace(-5, 5, 400) #valores en un rango 
        y = m*x+c
        limpiar()
        ax.plot(x,y,color='green')
        salidaActual = np.array([])
        for i in range(len(entradas)):
            z = w1*entradas[i,0]+w2*entradas[i,1]+bias
            if z <= 0:
                salidaActual=np.append(salidaActual,0)
                ax.plot(entradas[i,0],entradas[i,1],'Pg')
            else:
                salidaActual=np.append(salidaActual,1)
                ax.plot(entradas[i,0],entradas[i,1],'Pb')
        error = salidaDeseada - salidaActual
        print("Error",error)
        canvas.draw()
        #si hubo error
        if np.any(error):
            for i in range(len(entradas)):
                bias = bias + etha*error[i]
                w1 = w1 + etha*error[i]*entradas[i,0]
                #w1 = w1 + etha*error[i]*entradas[i,1]
                #w2 = w2 + etha*error[i]*entradas[i,0]
                w2 = w2 + etha*error[i]*entradas[i,1]
                B.set(str(round(bias,2)))
                W1.set(str(round(w1,2)))
                W2.set(str(round(w2,2)))
        else:
            break
        epocas += 1
    print(error)


def limpiar():
    ax.clear()
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)

    #Generar ejes de coordenadas
    ax.axhline(0, color='black',linewidth=1)
    ax.axvline(0, color='black',linewidth=1)
    canvas.draw()

def quit():
    ventana.quit()
    ventana.destroy()


ventana = ctk.CTk()
ventana.title("Practica 1")

#Titulo
frameTitle = ctk.CTkFrame(ventana)
Title = ctk.CTkLabel(frameTitle,text="PERCEPTRON",padx=300,font=("Arial",25),pady=10)
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
W1.set(str(round(w1,2)))
TW1 = ctk.CTkLabel(frameInputs,textvariable=W1)
TW1.grid(row=1,column=1,pady=20)

#W2
LW2 = ctk.CTkLabel(frameInputs,text="W2",padx=10)
LW2.grid(row=2,column=0)
W2 = ctk.StringVar()
W2.set(str(round(w2,2)))
TW2 = ctk.CTkLabel(frameInputs,textvariable=W2)
TW2.grid(row=2,column=1)

#BIAS
LB = ctk.CTkLabel(frameInputs,text="BIAS",padx=10)
LB.grid(row=3,column=0)
B = ctk.StringVar()
B.set(str(round(bias,2)))
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

#EPOCA ACTUAL
LEA = ctk.CTkLabel(frameInputs,text="EPOCA: ",padx=10)
LEA.grid(row=6,column=0)
EA = ctk.StringVar()
EA.set(str(round(ea,2)))
TEA = ctk.CTkLabel(frameInputs,textvariable=EA)
TEA.grid(row=6,column=1,pady=20)

#boton
btn= ctk.CTkButton(frameInputs,text = 'PLOT', command = Threading)
btn.grid(row=7,column=1,pady=20,padx=20)

#boton
btn2= ctk.CTkButton(frameInputs,text = 'Datos Adicionales', command = PuntosExtras)
btn2.grid(row=8,column=1,pady=20,padx=20)

frameInputs.pack(side='right',fill=ctk.BOTH,expand=True)

#INTERFAZ GRAFICA
frame = ctk.CTkFrame(ventana)
frame.pack(side='left')

generarPlano(frame)
ventana.protocol("WM_DELETE_WINDOW", quit)
ventana.mainloop()