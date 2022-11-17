from re import X
import tkinter as tk
from tkinter import filedialog
import os
from os import listdir
from PIL import Image, ImageTk
from pathlib import Path
#import mouse
#from pynput import mouse #pip install pynput
import cv2 #pip install opencv-python


#
# Ciência da Computação PUC Minas
# Campus Coração Eucarístico
#
# Trabalho Final de Processamento e Análise de Imagens
# Primeira Entrega
#
# Iago Morgado - 618090
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

#################################### FUNÇÕES ####################################

def comando():
    pass


def buscarUltimoRecorte():

    metodo = cv2.TM_CCORR_NORMED # escolhe o metodo de correlacao (correlação cruzada normalizada)

    # procura a variavel imgCorte na memória
    try:
        imgCorte
    except NameError: # se ela não existir, abre a imagem "crop.png" na raiz
        recorte = cv2.imread("crop.png")
    else:
        recorte = imgCorte.copy() #se ela existir, pega ela da memória

    arquivo = file
    imagem = cv2.imread(arquivo)    

    # usa o opencv para buscar a ocorrência mais próxima do recorte na imagem aberta
    # usando a correlação escolhida 
    busca = cv2.matchTemplate(recorte, imagem, metodo)

    # queremos a similaridade numerica e as coordenadas do minimo local
    min,max,mnLocCoord,mxLocCoord = cv2.minMaxLoc(busca)

    print(cv2.minMaxLoc(busca))

    # extrai as cooredenadas do melhor match
    tx,ty = mxLocCoord

    # recupera tamanho do recorte
    tlin,tcol = recorte.shape[:2]

    # desenha o retangulo na imagem grande (escolhe a cor do retangulo dependendo da similaridade aproximada)
    if (max == 1):
        cv2.rectangle(imagem, (tx,ty),(tx+tcol,ty+tlin),(0, 255, 0),2)
    elif max >= 0.99:
        cv2.rectangle(imagem, (tx,ty),(tx+tcol,ty+tlin),(0, 255, 255),2)
    else:
        cv2.rectangle(imagem, (tx,ty),(tx+tcol,ty+tlin),(0, 0, 255),2)
    
    # mostra a imagem grande com o match em volta
    cv2.imshow('Resultado',imagem)
    cv2.waitKey(0)


def buscarRecorte():
    # perguntar qual arquivo
    recorte = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha a imagem do recorte", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))

    metodo = cv2.TM_CCORR_NORMED # escolhe o metodo de correlacao (correlação cruzada normalizada)

    arquivo = file
    imagem = cv2.imread(arquivo)
    recorte = cv2.imread(recorte)

    # usa o opencv para buscar a ocorrência mais próxima do recorte na imagem aberta
    # usando a correlação escolhida 
    busca = cv2.matchTemplate(recorte, imagem, metodo)

    # queremos a similaridade numerica e as coordenadas do minimo local
    min,max,mnLocCoord,mxLocCoord = cv2.minMaxLoc(busca)

    print(cv2.minMaxLoc(busca))

    # extrai as cooredenadas do melhor match
    tx,ty = mxLocCoord

    # recupera tamanho do recorte
    tlin,tcol = recorte.shape[:2]

    # desenha o retangulo na imagem grande (escolhe a cor do retangulo dependendo da similaridade aproximada)
    if (max == 1):
        cv2.rectangle(imagem, (tx,ty),(tx+tcol,ty+tlin),(0, 255, 0),2)
    elif max >= 0.99:
        cv2.rectangle(imagem, (tx,ty),(tx+tcol,ty+tlin),(0, 255, 255),2)
    else:
        cv2.rectangle(imagem, (tx,ty),(tx+tcol,ty+tlin),(0, 0, 255),2)
    
    # mostra a imagem grande com o match em volta
    cv2.imshow('Resultado',imagem)
    cv2.waitKey(0)


def abrir_imagem():
    global file #!!

    file = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha a imagem", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))

    if file:
        img = Image.open(file)
        tam, tam = img.size # recuperando tamanho da imagem para seu posicionamento na interface
        img = ImageTk.PhotoImage(img)
        imagem_label.configure(image=img)
        imagem_label.image=img
        imagem_label.grid(column = 1, row = 0)

        # centraliza a imagem na hora de colocar ela
        if tam == 224:
            imagem_label.place(x=86, y=80) # imagem 224x224
        elif tam == 299:
            imagem_label.place(x=49, y=43) # imagem 299x299
        else:
            imagem_label.place(x=10, y = 10) # imagem esquisita fora do padrão


coord = [] # armazena as coordenadas de corte da imagem
def recorte_imagem(event, x, y, flags, param):
	global coord #!!

	if event == cv2.EVENT_LBUTTONDOWN: # se o botão esquerdo do mouse foi segurado, x e y iniciais são gravados
		coord = [(x, y)]
	elif event == cv2.EVENT_LBUTTONUP: # verifica se soltou o clique do botão esquerdo do mouse, x e y finais são gravados
		coord.append((x, y))		
		cv2.rectangle(imgCrop, coord[0], coord[1], 0, 2) # desenha um retângulo na região marcada de acordo com (x, y) inciais e finais 


def recorte():
    global imgCrop #!!
    global imgCorte #!!
    imgCrop = cv2.imread(file)
    clone = imgCrop.copy() # um clone da imagem original é criado para que possamos mudar a região de corte se necessário, 
                           # e a imagem original estará sem alterações pois as manipulações de corte ocorrerão no clone
    cv2.namedWindow("Recorte de Imagem") # Label para identifação da tela de recorte
    cv2.setMouseCallback("Recorte de Imagem", recorte_imagem) # Função de "escutar" o mouse

    while True:
        cv2.imshow("Recorte de Imagem", imgCrop) # mostra a imagem e espera a tecla correta para recortar
        key = cv2.waitKey(1)

        if key == ord("0"): # apertando "0" no teclado, o corte é resetado e outro pode ser escolhido
            imgCrop = clone.copy() # para recuperar a imagem sem o retângulo, recuperamos pelo clone
        elif key == ord("1"): # apertando "1" no teclado, a imagem é cortada
            break

    corte = clone[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]] # corte é feito na imagem clone a partir das coordenadas armazenadas

    imgCorte = corte.copy()

    cv2.imshow("Corte", corte)
    cv2.waitKey(0)

    cv2.imwrite('crop.png', corte) # imagem cortada é salva no diretório local


#################################### TREINO  ######################################


def escolher_caminho():

    global caminho_pasta #!!

    file_caminho = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha a imagem", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))

    #file_caminho = file_caminho.replace("\\", "/")
    result = Path(file_caminho).parent.parent.parent.parent

    caminho_pasta = str(result)

    #caminho_pasta = caminho_pasta.replace("\\", "/")
    print(caminho_pasta)


def treinar_classificador():

    print("dentro do classificador")
    print(caminho_pasta)

    treino244 = caminho_pasta + "\\kneeKL244\\test\\"
    treino299 = caminho_pasta + "\\kneeKL299\\test\\"

    for i in range(5):
        pasta = treino244 + str(i)
        print(pasta)

        caminho = repr(pasta)[1:-1]

        for imagem in os.listdir(caminho):
            if (imagem.endswith(".png")):
                print(imagem)


#################################### INTERFACE ####################################

menu = tk.Menu(root)
root.config(menu = menu)

#ler
file_menu = tk.Menu(menu)
menu.add_cascade(label="Ler", menu=file_menu)
file_menu.add_command(label="Imagem", command=abrir_imagem)
file_menu.add_command(label="Diretório", command=comando)

#processamento
process_menu = tk.Menu(menu)
menu.add_cascade(label="Processamento", menu=process_menu)
process_menu.add_command(label="Recortar e salvar recorte na raiz", command=recorte)
process_menu.add_command(label="Buscar do último recorte", command=buscarUltimoRecorte)
process_menu.add_command(label="Buscar de um arquivo", command=buscarRecorte)

#classificadores
classif_menu = tk.Menu(menu)
menu.add_cascade(label="Classificadores", menu=classif_menu)
classif_menu.add_command(label="Escolher Caminho a partir de imagem", command=escolher_caminho)
classif_menu.add_command(label="Escolher Classificador", command=comando)
classif_menu.add_command(label="Treinar Classificador", command=treinar_classificador)

#imagem inicial
imagem = Image.open('vazio.png')
imagem = ImageTk.PhotoImage(imagem)
imagem_label = tk.Label(image=imagem)
imagem_label.image = imagem
imagem_label.grid(column = 1, row = 0)
imagem_label.place(x=86, y=80)

root.mainloop()