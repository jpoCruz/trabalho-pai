import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from pathlib import Path

categories=['0','1','2','3','4']
data=[]

def make_data(data_dir,data_dir_2):
        for category in categories:
                path = os.path.join(data_dir,category)
                path_2=os.path.join(data_dir_2,category)
                label=categories.index(category)

                for img_name in os.listdir(path):
                        image_path=os.path.join(path,img_name)
                        image=cv2.imread(image_path)
                        try:
                                image=cv2.resize(image,(224,224))
                                image=np.array(image)
                                data.append([image,label])
                        except Exception as e:
                                pass
                for img_name_2 in os.listdir(path_2):
                        image_path_2=os.path.join(path_2,img_name_2)
                        image_2=cv2.imread(image_path_2)
                        try:
                                image_2=cv2.resize(image_2,(224,224))
                                image_2=np.array(image_2)
                                data.append([image_2,label])
                        except Exception as e:
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
                pick.close()

        feature=[]
        labels=[]

        for img, label in data_:
                feature.append(img)
                labels.append(label)

        feature=np.array(feature)
        labels=np.array(labels)

        return[feature,labels]
