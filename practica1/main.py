import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Definir estilos custom tkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


entradas = np.empty((0,2),dtype=float)



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
TW1 = ctk.CTkEntry(frameInputs,textvariable=W1)
TW1.grid(row=1,column=1,pady=20)

#W2
LW2 = ctk.CTkLabel(frameInputs,text="W2",padx=10)
LW2.grid(row=2,column=0)
W2 = ctk.StringVar()
TW2 = ctk.CTkEntry(frameInputs,textvariable=W2)
TW2.grid(row=2,column=1)

#BIAS
LB = ctk.CTkLabel(frameInputs,text="BIAS",padx=10)
LB.grid(row=3,column=0)
B = ctk.StringVar()
TB = ctk.CTkEntry(frameInputs,textvariable=B)
TB.grid(row=3,column=1,pady=20,padx=20)

#boton
btn= ctk.CTkButton(frameInputs,text = 'PLOT', command = Clasificar)
btn.grid(row=4,column=1,pady=20,padx=20)

frameInputs.pack(side='right',fill=ctk.BOTH,expand=True)

#INTERFAZ GRAFICA
frame = ctk.CTkFrame(ventana)
frame.pack(side='left')

generarPlano(frame)
ventana.protocol("WM_DELETE_WINDOW", quit)
ventana.mainloop()