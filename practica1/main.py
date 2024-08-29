import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

entradas = np.empty((0,2),dtype=float)



def generarPlano(frame):
    global ax,canvas
    figura = plt.figure(figsize=(5,5),dpi=100,facecolor='lightskyblue')
    ax = figura.add_subplot(111)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)

    #Generar ejes de coordenadas
    ax.axhline(0, color='black',linewidth=1)
    ax.axvline(0, color='black',linewidth=1)

    canvas = FigureCanvasTkAgg(figura, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    #CONECTAR CON FUNCION DE EVENTO
    canvas.mpl_connect('button_press_event',onclick)
    
def onclick(event):
    global entradas
    x = event.xdata
    y = event.ydata

    #comprobar que no se salga del limite del plano
    if x != None and y != None:
        ndatos = np.array([[x,y]])
        entradas = np.vstack((entradas,ndatos))
        ax.plot(x,y,'Pk')
        canvas.draw()
        print(entradas)


def Clasificar():
    global entradas
    w1 = ''
    w2 = ''
    bias = ''
    try:
        w1 = float(W1.get())
        w2 = float(W2.get())
        bias = float(B.get())
    except:
        print("Algo ocurrio con las entradas")
    #Seguro
    if w1 == '' or w2 == '' or bias == '':
        return
    
    m = -w1/w2
    c = -bias/w2
    x = np.linspace(-5, 5, 400) #valores en un rango 
    y = m*x+c
    limpiar()
    ax.plot(x,y,color='green')
    canvas.draw()


    for i in range(len(entradas)):
        z = w1*entradas[i,0]+w2*entradas[i,1]+bias
        if z <= 0:
            ax.plot(entradas[i,0],entradas[i,1],'Pr')
        else:
            ax.plot(entradas[i,0],entradas[i,1],'Pb')
        
    canvas.draw()
    

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


ventana = tk.Tk()
ventana.title("Practica 1")


#INTERFAZ ENTRADAS
frameInputs = tk.Frame(ventana,bg='lightblue')

TITULO = tk.Label(frameInputs,text="DEFINIR VALORES",padx=10,bg='lightblue')
TITULO.grid(row=0,column=1)

#W1
LW1 = tk.Label(frameInputs,text="W1",padx=10,bg='lightblue')
LW1.grid(row=1,column=0)
W1 = tk.StringVar()
TW1 = tk.Entry(frameInputs,textvariable=W1)
TW1.grid(row=1,column=1,pady=20)

#W2
LW2 = tk.Label(frameInputs,text="W2",padx=10,bg='lightblue')
LW2.grid(row=2,column=0)
W2 = tk.StringVar()
TW2 = tk.Entry(frameInputs,textvariable=W2)
TW2.grid(row=2,column=1)

#BIAS
LB = tk.Label(frameInputs,text="BIAS",padx=10,bg='lightblue')
LB.grid(row=3,column=0)
B = tk.StringVar()
TB = tk.Entry(frameInputs,textvariable=B)
TB.grid(row=3,column=1,pady=20,padx=20)

#boton
btn=tk.Button(frameInputs,text = 'PLOT', command = Clasificar)
btn.grid(row=4,column=1,pady=20,padx=20)

frameInputs.pack(side='right',fill=tk.BOTH,expand=True)

#INTERFAZ GRAFICA
frame = tk.Frame(ventana)
frame.pack(side='left')

generarPlano(frame)
ventana.protocol("WM_DELETE_WINDOW", quit)
ventana.mainloop()