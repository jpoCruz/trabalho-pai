import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from skimage import filters
from tkinter import filedialog
from pathlib import Path
import tkinter as tk
import time

#
# Ciência da Computação PUC Minas
# Campus Coração Eucarístico
#
# Trabalho Final de Processamento e Análise de Imagens
# Entrega Final
#
# Iago Morgado - 618090
# João Paulo Oliveira Cruz - 615932
# Pedro Rodrigues - 594451
#


def popupInfo(message): #abre um popup com a string "message" como corpo
    print("[!] Abrindo popup de métricas")

    pop = tk.Toplevel()
    pop.title("Métricas")
    pop.geometry("330x380")
    pop.config(bg="#ffffff")
    corpo = tk.Label(pop, text=message) #usa uma label para escrever o conteúdo do popup
    button1 = tk.Button(pop, text="Ok", command = pop.destroy) #botão para fechar o popup
    corpo.pack()
    button1.pack()

    pop.mainloop()

def popupSmall(message): #abre um popup com a string "message" como corpo
    print("[!] Abrindo popup de predição!")

    pop = tk.Toplevel()
    pop.title("Predição")
    pop.geometry("330x120")
    pop.config(bg="#ffffff")
    corpo = tk.Label(pop, text=message) #usa uma label para escrever o conteúdo do popup
    button1 = tk.Button(pop, text="Ok", command = pop.destroy) #botão para fechar o popup
    corpo.pack()
    button1.pack()

    pop.mainloop()


def treinoSVM():

    start_time = time.time()

    ####### Treino #######
    categories = ['0', '1', '2', '3', '4'] #categorias de osteoartrite
    data = [] #dados para o dataset

    #escolhendo pastas de treino

    file_caminho_dir1 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de treino", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir1 = Path(file_caminho_dir1).parent.parent #escolhendo a imagem, recebe-se a pasta de categorias de imagens (224, 224)

    file_caminho_dir2 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de treino", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir2 = Path(file_caminho_dir2).parent.parent #escolhendo a imagem, recebe-se a pasta de categorias (299, 299)

    
    #Navegação por todas as imagens (224, 224) de todas as pastas
    #dir1
    for category in categories:
        path = os.path.join(file_caminho_dir1, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img) #caminho da imagem
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) #imagem lida em tons de cinza

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224)) #redimencionar imagem para o corte preciso
                img_joelho = img_joelho[73:166, 3:222] #corte da articulação na parte do meio


                img_joelho = cv2.equalizeHist(img_joelho)# equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1]# inversão da imagem

                # extração de características(borda)
                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                #transformação de imagens em um array 1D 
                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()

                #características da imagem com o respectivo label adicionado no dataset
                data.append([image_joelho_data, label])
                data.append([image_joelho_invertido_data, label])
            except Exception as e:
                pass

    #Navegação por todas as imagens (299, 299) de todas as pastas
    #dir2
    for category in categories:
        path = os.path.join(file_caminho_dir2, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img) #caminho da imagem
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) #imagem lida em tons de cinza

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224)) #redimencionar imagem para o corte preciso
                img_joelho = img_joelho[73:166, 3:222] #corte da articulação na parte do meio

                img_joelho = cv2.equalizeHist(img_joelho) #equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1] #inversão da imagem

                #extração de características(borda)
                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                #transformação de imagens em um array 1D 
                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()

                #características da imagem com o respectivo label adicionado no dataset
                data.append([image_joelho_data, label])
                data.append([image_joelho_invertido_data, label])
            except Exception as e:
                pass


    random.shuffle(data) #mistura dos dados para ajudar no treino

    features = [] #características dos dados de treino
    labels = [] #labels dos dados de treino

    #separação de características e labels para treinar o modelo
    for feature, label in data:
        features.append(feature)
        labels.append(label)

    #treinamento do modelo
    model = SVC(C = 10, kernel = 'poly', gamma = 35)
    model.fit(features, labels)
    
    #salvando modelo externamente
    pick_model = open('model.h5', 'wb')
    pickle.dump(model, pick_model)
    pick_model.close()
    

    ########## TESTE ##########

    #escolhendo pastas de teste
    file_caminho_dir1 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de teste", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir1 = Path(file_caminho_dir1).parent.parent #escolhendo a imagem, recebe-se a pasta de categorias de imagens (224, 224)

    file_caminho_dir2 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de teste", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir2 = Path(file_caminho_dir2).parent.parent #escolhendo a imagem, recebe-se a pasta de categorias de imagens (299, 299)

    features_teste = [] #características dos dados de teste
    labels_teste = [] #labels dos dados de teste

    #Navegação por todas as imagens (224, 224) de todas as pastas
    #dir 1
    for category in categories:
        path = os.path.join(file_caminho_dir1, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img) #caminho da imagem
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) #imagem lida em tons de cinza

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224)) #redimencionar imagem para o corte preciso
                img_joelho = img_joelho[73:166, 3:222] #corte da articulação na parte do meio


                img_joelho = cv2.equalizeHist(img_joelho)# equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1]# inversão da imagem

                #extração de características(borda)
                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                #transformação de imagens em um array 1D
                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()           
                

                #características da imagem com o respectivo label adicionado no dataset
                features_teste.append(image_joelho_data)
                labels_teste.append(label)
                features_teste.append(image_joelho_invertido_data)
                labels_teste.append(label)
            except Exception as e:
                pass

    #Navegação por todas as imagens (299, 299) de todas as pastas
    #dir2
    for category in categories:
        path = os.path.join(file_caminho_dir2, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img) #caminho da imagem
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) #imagem lida em tons de cinza

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224)) #redimencionar imagem para o corte preciso
                img_joelho = img_joelho[73:166, 3:222] #corte da articulação na parte do meio


                img_joelho = cv2.equalizeHist(img_joelho) #equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1] #inversão da imagem

                #extração de características(borda)
                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                #transformação de imagens em um array 1D
                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()           
                
                #características da imagem com o respectivo label adicionado no dataset
                features_teste.append(image_joelho_data)
                labels_teste.append(label)
                features_teste.append(image_joelho_invertido_data)
                labels_teste.append(label)
            except Exception as e:
                pass

    #separação dos dados de treino
    X_train, X_test, y_train, y_test = train_test_split(features_teste, labels_teste, test_size = 0.01)

    #modelo sendo carregado
    pick_model_treinado = open('model.h5', 'rb')
    model = pickle.load(pick_model_treinado)
    pick_model_treinado.close()

    #métricas do modelo
    prediction = model.predict(X_test)

    report = metrics.classification_report(y_test, prediction)
    cmatrix = confusion_matrix(y_test, prediction)
    acuracia = model.score(X_test, y_test)
    segundos = (time.time() - start_time)
    mensagem = ("\nTempo de execução: " + segundos + "s\n" + report + "\nMatriz de confusão:\n" + str(cmatrix) + "\nAcurácia: " + str(acuracia) + "\n")

    popupInfo(mensagem)



def classificarSVM(file):

    start_time = time.time()

    img = cv2.imread(file, 0) #imagem lida em tons de cinza
    img_recortada = img[73:166, 3:222] #imagem é recortada no meio
    img_recortada = cv2.equalizeHist(img_recortada) #imagem é equalizada
    img_recortada = filters.sobel(img_recortada) #aplicado filtro de sobel

    #modelo sendo carregado
    pick_model_treinado = open('model.h5', 'rb')
    model = pickle.load(pick_model_treinado)
    pick_model_treinado.close()

    #predição de classe da imagem selecionada 
    prediction = model.predict(img_recortada.reshape(1, -1))
    segundos = (time.time() - start_time)

    mensagem = "\nTempo de predição: " + segundos + "s\n" + str(prediction) + "]\n"

    popupSmall(mensagem)





def treinoSVM_Binario():

    start_time = time.time()

    #####Treino#####

    categories = ['0', '1', '2', '3', '4'] #categorias de osteoartrite
    data = [] #dados para o dataset

    
    #escolhendo pastas de treino
    file_caminho_dir1 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de treino", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir1 = Path(file_caminho_dir1).parent.parent #escolhendo a imagem, recebe-se a pasta de categorias de imagens (224, 224)

    file_caminho_dir2 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de treino", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir2 = Path(file_caminho_dir2).parent.parent #escolhendo a imagem, recebe-se a pasta de categorias de imagens (299, 299)

    
    #Navegação por todas as imagens (224, 244) de todas as pastas
    #dir1
    for category in categories:
        path = os.path.join(file_caminho_dir1, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img) #caminho da imagem
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) #leitura da imagem em tons de cinza

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224)) #redimencionar imagem para o corte preciso
                img_joelho = img_joelho[73:166, 3:222] #corte da articulação na parte do meio


                img_joelho = cv2.equalizeHist(img_joelho)# equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1] #inversão da imagem

                #extração de características(borda)
                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                #transformação de imagens em um array 1D
                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()

                #características da imagem com o respectivo label adicionado no dataset
                data.append([image_joelho_data, label])
                data.append([image_joelho_invertido_data, label])
            except Exception as e:
                pass

    #Navegação por todas as imagens (299, 299) de todas as pastas
    #dir2
    for category in categories:
        path = os.path.join(file_caminho_dir2, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img) #caminho da imagem
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) #leitura da imagem em tons de cinza

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224)) #redimencionar imagem para o corte preciso
                img_joelho = img_joelho[73:166, 3:222] #corte da articulação na parte do meio

                img_joelho = cv2.equalizeHist(img_joelho)# equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1]# inversão da imagem

                #extração de características(borda)
                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                #transformação de imagens em um array 1D
                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()

                #características da imagem com o respectivo label adicionado no dataset
                data.append([image_joelho_data, label])
                data.append([image_joelho_invertido_data, label])
            except Exception as e:
                pass

    random.shuffle(data) #mistura dos dados para ajudar no treino

    features = [] #características dos dados de treino
    labels = [] #labels dos dados de treino

    #separação de características e labels para treinar o modelo
    for feature, label in data:
        features.append(feature)

        #para classificação binária, caso a classe for '2', '3' ou '4', indica que possui osteoartrite, do contrário não
        if label > 1:
            labels.append(1)
        else:
            labels.append(0)

    #treinamento do modelo
    model = SVC(C = 5, kernel = 'poly', gamma = 0.5)
    model.fit(features, labels)
    
    #salvando modelo externamente
    pick_model = open('model_binario.h5', 'wb')
    pickle.dump(model, pick_model)
    pick_model.close()
    

    #####Teste#####

    #escolhendo pastas de teste
    file_caminho_dir1 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de teste", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir1 = Path(file_caminho_dir1).parent.parent #escolhendo a imagem, recebe-se a pasta de categorias de imagens (224, 224)

    file_caminho_dir2 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de teste", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir2 = Path(file_caminho_dir2).parent.parent #escolhendo a imagem, recebe-se a pasta de categorias de imagens (299, 299)

    features_teste = [] #características das imagens de teste
    labels_teste = [] #labels das imagens de teste

    #Navegação por todas as imagens (224, 224) de todas as pastas
    #dir 1
    for category in categories:
        path = os.path.join(file_caminho_dir1, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img) #caminho da imagem
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) #imagem lida em tons de cinza

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224)) #redimencionar imagem para corte preciso
                img_joelho = img_joelho[73:166, 3:222]# corte da articulação na parte do meio


                img_joelho = cv2.equalizeHist(img_joelho)# equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1]# inversão da imagem

                #extração de características(borda)
                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                #transformação de imagens em array 1D
                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()           
                


                #divisão dos dados de características e seus respectivos labels seguindo a regra binária
                features_teste.append(image_joelho_data)
                if label > 1:
                    labels_teste.append(1)
                else: 
                    labels_teste.append(0)


                features_teste.append(image_joelho_invertido_data)
                if label > 1:
                    labels_teste.append(1)
                else: 
                    labels_teste.append(0)
            except Exception as e:
                pass

    #Navegação por todas as imagens (299, 299) de todas as pastas
    #dir2
    for category in categories:
        path = os.path.join(file_caminho_dir2, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img) #caminho da imagem
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) #leitura da imagem em tons de cinza

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224)) #redimencionar a imagem para corte preciso
                img_joelho = img_joelho[73:166, 3:222]# corte da articulação na parte do meio


                img_joelho = cv2.equalizeHist(img_joelho) #equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1] #inversão da imagem

                #extração de características(borda)
                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                #transformação de imagens em array 1D
                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()           
                
                
                
                #divisão dos dados de características e seus respectivos labels seguindo a regra binária
                features_teste.append(image_joelho_data)
                if label > 1:
                    labels_teste.append(1)
                else: 
                    labels_teste.append(0)


                features_teste.append(image_joelho_invertido_data)
                if label > 1:
                    labels_teste.append(1)
                else: 
                    labels_teste.append(0)
            except Exception as e:
                pass

        
    #separação dos dados de treino
    X_train, X_test, y_train, y_test = train_test_split(features_teste, labels_teste, test_size = 0.01)


    #modelo sendo carregado
    pick_model_treinado = open('model_binario.h5', 'rb')
    model = pickle.load(pick_model_treinado)
    pick_model_treinado.close()

    #métricas do modelo
    prediction = model.predict(X_test)

    report = metrics.classification_report(y_test, prediction)
    cmatrix = confusion_matrix(y_test, prediction)
    acuracia = model.score(X_test, y_test)
    specifity = cmatrix[1, 1]/(cmatrix[1, 0]+cmatrix[1, 1])
    segundos = (time.time() - start_time)

    mensagem = ("\nTempo de execução: " + segundos + "s\n" + report + "\nMatriz de confusão:\n" + str(cmatrix) + "\nAcurácia: " + str(acuracia) + "\nEspecificidade: " + str(specifity) + "\n")

    popupInfo(mensagem)






def classificarSVM_Binario(file):

    start_time = time.time()
        
    img = cv2.imread(file, 0) #imagem lida em tons de cinza
    img_recortada = img[73:166, 3:222] #imagem é recortada no meio
    img_recortada = cv2.equalizeHist(img_recortada) #imagem é equalizada
    img_recortada = filters.sobel(img_recortada) #aplicado filtro de sobel

    #modelo sendo carregado
    pick_model_treinado = open('model_binario.h5', 'rb')
    model = pickle.load(pick_model_treinado)
    pick_model_treinado.close()

    #predição de classe da imagem selecionada 
    prediction = model.predict(img_recortada.reshape(1, -1))
    segundos = (time.time() - start_time)

    mensagem = "\nTempo de predição: " + segundos + "s\n" + "\nClasse [" + str(prediction) + "]\n"

    popupSmall(mensagem)
