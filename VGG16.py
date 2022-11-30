import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses

def getModel():
    input_layer = tf.keras.layers.Input([224,224,1])
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
    fc6 = tf.keras.layers.Dense(units=256, name='fc6', activation='relu')(flatten)
    fc7 = tf.keras.layers.Dense(units=128, name='fc7', activation='relu')(fc6)
    fc8 = tf.keras.layers.Dense(units=5, name='fc8',activation=None)(fc7)

    prob = tf.nn.softmax(fc8)

    model = tf.keras.Model(input_layer, prob)
    model.summary()

    return model

def trainVGG(x_train, y_train, x_test):
    #x_train, x_test = tf.cast(x_train,tf.float32), tf.cast(x_test, tf.float32)

    train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train))
    train_dataset=train_dataset.batch(batch_size=10)

    model=getModel()
    loss_object= tf.keras.losses.SparseCategoricalCrossentropy()
    optmizer=tf.keras.optimizers.Adam()

    train_loss=tf.keras.metrics.Mean(name='train_loss')
    train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    @tf.function
    def train_step(images,labels):
        with tf.GradientTape() as tape:
            predictions = model(images,training= True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients=tape.gradient(loss, model.trainable_variables)
        optmizer.apply_gradients(grads_and_vars=zip(gradients,model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        
    print(":::CONTROLE:::") 
    for epoch in range(1):
        train_loss.reset_states()
        train_accuracy.reset_states()
        step = 0
        for images, labels in train_dataset:
            step+=1 
            train_step(images, labels)
            if step%10 ==0:
                print('=> epoch: %i, loss: %.4f, train_accuracy: %.4f'%(epoch+1,train_loss.result(), train_accuracy.result()))
    print(":::CONTROLE:::")
    model.save('VGG16.h5')

def testVGG(x_test,y_test):
    categories=['0','1','2','3','4']

    model= tf.keras.models.load_model('VGG16.h5')
    prediction = model(x_test[0:9])

    plt.figure(figsize=(8,8))

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(x_test[i])
        plt.xlabel('Pedicted:%s\n Actual: %s'%(categories[np.argmax(prediction[i])],
            categories[y_test[i]]))

        plt.xticks([])

    plt.show()