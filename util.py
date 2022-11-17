import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

categories=['0','1','2','3','4']
data =[]

def make_train_data():
        data_dir='./KneeXrayData/KneeXrayData/ClsKLData/kneeKL224/train'
        for category in categories:
                path = os.path.join(data_dir,category)
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
        print(len(data))                        
        pik = open('knee.pickle','wb')
        pickle.dump(data,pik)
        pik.close()

def load_data():
        make_train_data()
        pick=open('knee.pickle','rb')
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

#feature, label = load_data()
#plt.figure(figsize=(8,8))

#for i in range(25):
        #plt.subplot(5,5,i+1)
        #plt.imshow(feature[i])
        #plt.text(5,0,s=label[i])
        #plt.xticks([])

#plt.show()