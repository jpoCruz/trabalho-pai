import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import random
from pathlib import Path

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

categories=['0','1','2','3','4']
# Função que cria os arquivos .pickle necessários para criar o dataset
def make_data(data_dir,data_dir_2):
        data=[]
        for category in categories:
                path = os.path.join(data_dir,category)
                path_2=os.path.join(data_dir_2,category)
                label=categories.index(category)

                for img_name in os.listdir(path):
                        image_path=os.path.join(path,img_name)
                        image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)# leitura da imagem em tons de cinza
                        try:    
                                image=cv2.resize(image,(32,32))

                                image = cv2.equalizeHist(image)# equalização de histograma da imagem
                                image_invertida = image[:, ::-1]# inversão da imagem

                                image = np.array(image)
                                image_invertida = np.array(image_invertida)

                                data.append([image,label])
                                data.append([image_invertida,label])
                        except Exception as e:
                                print(e)
                                pass
                              
                for img_name_2 in os.listdir(path_2):
                        image_path_2=os.path.join(path_2,img_name_2)
                        image_2=cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)# leitura da imagem em tons de cinza
                        try:
                                image_2=cv2.resize(image_2,(32,32))

                                image_2 = cv2.equalizeHist(image_2)# equalização de histograma da imagem
                                image_invertida2 = image_2[:, ::-1]# inversão da imagem

                                image_2 = np.array(image_2)
                                image_invertida2 = np.array(image_invertida2)

                                data.append([image_2,label])
                                data.append([image_invertida2, label])

                        except Exception as e:
                                print(e)
                                pass

        if(data_dir.stem=='train'):
                print(len(data))
                pik = open('knee_train.pickle','wb')
                pickle.dump(data,pik)
                pik.close()
        elif (data_dir.stem=='test'):
                print(len(data))
                pik = open('knee_test.pickle','wb')
                pickle.dump(data,pik)
                pik.close()
        elif(data_dir.stem=='val'):
                print(len(data))
                pik = open('knee_val.pickle','wb')
                pickle.dump(data,pik)
                pik.close()

#Função que carrega o dataset contido nos arquivos .pickle 
def load_data(data_dir,data_dir_2):
        make_data(data_dir,data_dir_2)
        
        if(data_dir.stem=='train'):
                pick = open('knee_train.pickle','rb')
        elif (data_dir.stem=='test'):
                pick = open('knee_test.pickle','rb')
        elif(data_dir.stem=='val'):
                pick = open('knee_val.pickle','rb')
        
        if(pick!=None):
                data_= pickle.load(pick)
                random.shuffle(data_)
                pick.close()

        feature=[]
        labels=[]

        for img, label in data_:
                feature.append(img)
                labels.append(label)

        feature = np.array(feature, dtype = np.float32)
        feature = feature/ 255.
        labels = np.array(labels)

        return[feature,labels]

#Função que define modelo VGG16. 
#Não mais utilizada na implementação final
def getModel():
    input_layer = tf.keras.layers.Input([32,32,3])
    #block1 
    conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3],strides=[1,1],padding='same',activation='relu', name='conv1_1')(input_layer)
    conv1_2 = tf.keras.layers.Conv2D(filters= 64, kernel_size=[3,3], strides= [1,1],padding='same',activation='relu', name='conv1_2')(conv1_1)
    pool1_1 = tf.nn.max_pool(conv1_2, ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME', name='pool1_1')

    #block 2
    conv2_1 = tf.keras.layers.Conv2D(filters= 128, kernel_size=[3,3],strides=[1,1],padding='same', activation='relu', name='conv2_1')(pool1_1)
    conv2_2 = tf.keras.layers.Conv2D(filters = 128, kernel_size=[3,3], strides = [1,1],padding='same', activation='relu', name='conv2_2')(conv2_1)
    pool2_1 = tf.nn.max_pool(conv2_2, ksize = [1,2,2,1], strides= [1,2,2,1],padding='SAME', name='pool2_1')

    #block3 
    conv3_1 = tf.keras.layers.Conv2D(filters= 256, kernel_size=[3,3], strides = [1,1],padding='same', activation='relu', name='conv3_1')(pool2_1)
    conv3_2 = tf.keras.layers.Conv2D(filters = 256, kernel_size=[3,3], strides=[1,1],padding='same', activation='relu', name='conv3_2')(conv3_1)
    conv3_3 = tf.keras.layers.Conv2D(filters = 256, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv3_3')(conv3_2)
    conv3_4 = tf.keras.layers.Conv2D(filters = 256, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv3_4')(conv3_3)
    pool3_1 = tf.nn.max_pool(conv3_4, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME",name='pool3_1')

    #block4 
    conv4_1 = tf.keras.layers.Conv2D(filters = 512, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv4_1')(pool3_1)
    conv4_2 = tf.keras.layers.Conv2D(filters = 512, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv4_2')(conv4_1)
    conv4_3 = tf.keras.layers.Conv2D(filters = 512, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv4_3')(conv4_2)
    conv4_4 = tf.keras.layers.Conv2D(filters = 512, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv4_4')(conv4_3)
    pool4_1 = tf.nn.max_pool(conv4_4, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name='pool4_1')

    #block5
    conv5_1 = tf.keras.layers.Conv2D(filters = 512, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv5_1')(pool4_1)
    conv5_2 = tf.keras.layers.Conv2D(filters = 512, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv5_2')(conv5_1)
    conv5_3 = tf.keras.layers.Conv2D(filters = 512, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv5_3')(conv5_2)
    conv5_4 = tf.keras.layers.Conv2D(filters = 512, kernel_size= [3,3], strides=[1,1],padding='same', activation='relu', name='conv5_4')(conv5_3)
    pool5_1 = tf.nn.max_pool(conv5_4, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME', name='pool5_1')

    flatten  = tf.keras.layers.Flatten()(pool5_1)
    fc6 = tf.keras.layers.Dense(units=4096, name='fc6', activation='relu')(flatten)
    fc7 = tf.keras.layers.Dense(units=4096, name='fc7', activation='relu')(fc6)
    fc8 = tf.keras.layers.Dense(units=5, name='fc8',activation=None)(fc7)

    prob = tf.nn.softmax(fc8)

    model = tf.keras.Model(input_layer, prob)

    return model

def trainVGG(x_train,x_test,y_train,y_test):
    model=tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (32,32,3)) #obtem o modelo VGG16 a partir da biblioteca do Tensorflow
    for layer in model.layers: #Define as camadas convolucionais como não treinaveis para preservar os parâmetros pré treinados do modelo
      layer.trainable = False
    #Define as camadas da rede totalmente conectada do modelo 
    x = tf.keras.layers.Flatten()(model.output)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(5, activation = 'softmax')(x)
    #Definição final do modelo com o input e output correto da rede
    model = tf.keras.Model(inputs = model.input, outputs = predictions)
    opt=tf.keras.optimizers.Adam()#Definição da função de otimização do treino
    loss=tf.keras.losses.SparseCategoricalCrossentropy()#Definição da função de perda do treino
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]#Definição das metricas do treino 

    model.compile(optimizer=opt,loss=loss,metrics=metrics)#Compilação do modelo com otimizador,função de perda e métricas
    #Treinamento do modelo, treino a teste são feitos simultaneamente
    #hist recebe o historico dos valores de perda e metricas do modelo
    hist = model.fit(x_train,y_train,batch_size=5,epochs=40,validation_data=(x_test,y_test))
    
    model.save('Vgg16.h5')#Salva o modelo treinado em um arquivo
    model.summary()#mostra a topologia da rede

    #mostra os valores de acurácia em cada epoca de treinamento do modelo em um gráfico
    plt.subplot(2, 1, 1)
    plt.title('Acurácia')
    plt.plot(hist.history['accuracy'], '-o', label='treino')
    plt.plot(hist.history['val_accuracy'], '-o', label='teste')
    plt.legend(loc='lower right')
    #mostra os valores de perda em cada epoca de treinamento do modelo em um gráfico
    plt.subplot(2, 1, 2)
    plt.title('Perda')
    plt.plot(hist.history['loss'], '-o', label='treino')
    plt.plot(hist.history['val_loss'], '-o', label='teste')
    plt.legend(loc='lower right')
    
    plt.gcf().set_size_inches(15, 12)