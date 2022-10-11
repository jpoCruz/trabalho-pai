import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import mouse
from pynput import mouse #pip install pynput


#
# Ciência da Computação PUC Minas
# Campus Coração Eucarístico
#
# Trabalho Final de Processamento e Análise de Imagens
#
# Iago - 
# João Paulo Oliveira Cruz - 615932
# Pedro Rodrigues - 594451
#

root = tk.Tk()
root.resizable(False,False)
root.title('Diagnóstico Automático de da Osteoartrite Femorotibial')
root.iconbitmap("icon.ico")
canvas = tk.Canvas(root, width=576, height=384)
canvas.grid(columnspan=3, rowspan=3)

info = tk.Frame(root, width=240, height=480, bg="#b3b3b3")
info.grid(row = 0, column=2)
info.place(x=400)

#Comandos
def comando():
    pass

def abrir_imagem():
    file = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha a imagem", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    if file:
        img = Image.open(file)
        tam, tam = img.size
        img = ImageTk.PhotoImage(img)
        imagem_label.configure(image=img)
        imagem_label.image=img
        imagem_label.grid(column = 1, row = 0)
        if tam == 224:
            imagem_label.place(x=86, y=80) # imagem 224x224
        else :
            imagem_label.place(x=49, y=43) # imagem 299x299
        

def recorte():
    with mouse.Listener(on_click=on_click) as coordenadas:
        coordenadas.join()
    with mouse.Listener(on_click=on_click) as coordenadas:
        coordenadas.join()
        
        

def on_click(x, y, button, pressed):
    if pressed:
        #mouse.get_position()
        print(x, y)
        return False


        

#Menu
menu = tk.Menu(root)
root.config(menu = menu)

#Ler
file_menu = tk.Menu(menu)
menu.add_cascade(label="Ler", menu=file_menu)
file_menu.add_command(label="Imagem", command=abrir_imagem)
file_menu.add_command(label="Diretório", command=comando)

#Processamento
process_menu = tk.Menu(menu)
menu.add_cascade(label="Processamento", menu=process_menu)
process_menu.add_command(label="Recortar e salvar como...", command=recorte)
process_menu.add_command(label="Buscar do último recorte", command=comando)
process_menu.add_command(label="Buscar de um arquivo", command=comando)

#Classificadores
classif_menu = tk.Menu(menu)
menu.add_cascade(label="Classificadores", menu=classif_menu)
classif_menu.add_command(label="Escolher Classificador", command=comando)
classif_menu.add_command(label="Treinar Classificador", command=comando)

#------------------------------------------------------------------------------------------#

#Imagem inicial
imagem = Image.open('vazio.png')
imagem = ImageTk.PhotoImage(imagem)
imagem_label = tk.Label(image=imagem)
imagem_label.image = imagem
imagem_label.grid(column = 1, row = 0)
imagem_label.place(x=86, y=80)


#def on_click(x, y, button, pressed):
    #if not pressed and button == mouse.Button.left:
       # print(x, y)

#with mouse.Listener(on_click=on_click) as coordenadas:
    #coordenadas.join()



root.mainloop()
