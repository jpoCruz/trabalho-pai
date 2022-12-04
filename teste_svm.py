import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from skimage import filters
from tkinter import filedialog
from pathlib import Path #para administrar caminhos

def testeSVM():
    file_caminho_dir1 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de teste", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir1 = Path(file_caminho_dir1).parent

    file_caminho_dir2 = filedialog.askopenfilename(initialdir=os.getcwd(), title = "Escolha uma imagem da pasta de teste", filetypes=(("PNG File", "*.png"), ("JPG File", "*.jpg"), ("All Files", "*.*")))
    file_caminho_dir2 = Path(file_caminho_dir2).parent

    categories = ['0', '1', '2', '3', '4']
    features = []
    labels = []

    #dir 1
    for category in categories:
        path = os.path.join(file_caminho_dir1, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224))
                img_joelho = img_joelho[73:166, 3:222]# corte da articulação


                img_joelho = cv2.equalizeHist(img_joelho)# equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1]# inversão da imagem

                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()           
                

                features.append(image_joelho_data)
                labels.append(label)
                features.append(image_joelho_invertido_data)
                labels.append(label)
            except Exception as e:
                pass

    #dir2
    for category in categories:
        path = os.path.join(file_caminho_dir2, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            img_joelho = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

            try:
                img_joelho = cv2.resize(img_joelho, (224, 224))
                img_joelho = img_joelho[73:166, 3:222]# corte da articulação


                img_joelho = cv2.equalizeHist(img_joelho)# equalização de histograma da imagem
                img_invertida = img_joelho[:, ::-1]# inversão da imagem

                img_joelho = filters.sobel(img_joelho)
                img_invertida = filters.sobel(img_invertida)

                image_joelho_data = np.array(img_joelho).flatten()
                image_joelho_invertido_data = np.array(img_invertida).flatten()           
                

                features.append(image_joelho_data)
                labels.append(label)
                features.append(image_joelho_invertido_data)
                labels.append(label)
            except Exception as e:
                pass

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.98)


    pick_model_treinado = open('model.h5', 'rb')
    model = pickle.load(pick_model_treinado)
    pick_model_treinado.close()

    prediction = model.predict(X_test)

    print("Prediction is: ", prediction)
    print(confusion_matrix(y_test, prediction))

    print("Acuracia: ", model.score(X_test, y_test))
