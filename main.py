import tkinter as tk
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

#
# Ciência da Computação PUC Minas
# Campus Coração Eucarístico
#
# Trabalho Final de Processamento e Análise de Imagens
#
# Iago - 
# João Paulo Oliveira Cruz - 615932
# Pedro -
#

root = tk.Tk()
root.resizable(False,False)
root.title('Diagnóstico Automático de da Osteoartrite Femorotibial')
root.iconbitmap("icon.ico")
canvas = tk.Canvas(root, width=576, height=384)
canvas.grid(columnspan=3, rowspan=3)

info = tk.Frame(root, width=240, height=480, bg="#b3b3b3")
info.grid(row = 0, column=2)
info.place(x=340)

#Comandos
def comando():
    pass

def abrir_imagem():
    file = askopenfile(parent=root, mode='rb',title="Escolha uma imagem", filetype=[("Image file", "*.png")])
    if file:
        pass #atualizar imagem na tela


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
process_menu.add_command(label="Recortar e salvar como...", command=comando)
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
imagem_label.place(x=55, y=75)


root.mainloop()