from re import X
import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import mouse
from pynput import mouse #pip install pynput
import cv2 #pip install opencv-python


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
    global file

    file = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha a imagem", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    if file:
        img = Image.open(file)
        tam, tam = img.size # Recuperando tamanho da imagem para seu posicionamento na interface
        img = ImageTk.PhotoImage(img)
        imagem_label.configure(image=img)
        imagem_label.image=img
        imagem_label.grid(column = 1, row = 0)
        if tam == 224:
            imagem_label.place(x=86, y=80) # Imagem 224x224
        else :
            imagem_label.place(x=49, y=43) # Imagem 299x299


coord = []# Armazena as coordenadas de corte da imagem
def recorte_imagem(event, x, y, flags, param):
	global coord

	if event == cv2.EVENT_LBUTTONDOWN:# Se o botão esquerdo do mouse foi segurado, x e y iniciais são gravados
		coord = [(x, y)]
	elif event == cv2.EVENT_LBUTTONUP:# Verifica se soltou o clique do botão esquerdo do mouse, x e y finais são gravados
		coord.append((x, y))		
		cv2.rectangle(imgCrop, coord[0], coord[1], 0, 2)# Desenha um retângulo na região marcada de acordo com (x, y) inciais e finais 

def recorte():
    global imgCrop
    imgCrop = cv2.imread(file)
    clone = imgCrop.copy()# Um clone da imagem original é criado para que possamos mudar a região de corte se necessário, 
                          # e a imagem original estará sem alterações pois as manipulações de corte ocorrerão no clone
    cv2.namedWindow("Recorte de Imagem")# Label para identifação da tela de recorte
    cv2.setMouseCallback("Recorte de Imagem", recorte_imagem)# Função de "escutar" o mouse

    while True:
        cv2.imshow("Recorte de Imagem", imgCrop)# Mostra a imagem e espera a tecla correta para recortar
        key = cv2.waitKey(1)

        if key == ord("0"):# Apertando "0" no teclado, o corte é resetado e outro pode ser escolhido
            imgCrop = clone.copy()# Para recuperar a imagem sem o retângulo, recuperamos pelo clone
        elif key == ord("1"):# Apertando "1" no teclado, a imagem é cortada
            break

    if len(coord) == 2:# Se houverem duas coordenadas e saimos do loop, a imagem foi cortada
        corte = clone[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]]# Corte é feito na imagem clone a partir das coordenadas armazenadas
        cv2.imshow("Corte", corte)
        cv2.waitKey(0);

    cv2.imwrite('crop.png', corte)# Imagem cortada é salva no diretório local


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

root.mainloop()
