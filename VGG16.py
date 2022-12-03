import tensorflow as tf
import matplotlib.pyplot as plt

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
    model=tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (32,32,3))
    for layer in model.layers:
      layer.trainable = False

    x = tf.keras.layers.Flatten()(model.output)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(5, activation = 'softmax')(x)

    model = tf.keras.Model(inputs = model.input, outputs = predictions)
    opt=tf.keras.optimizers.Adam()
    loss=tf.keras.losses.SparseCategoricalCrossentropy()
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]

    model.compile(optimizer=opt,loss=loss,metrics=metrics)

    hist = model.fit(x_train,y_train,batch_size=1,epochs=10,validation_data=(x_test,y_test))
    model.save('Vgg16.h5')
    model.summary()

    plt.subplot(2, 1, 1)
    plt.title('Acurácia')
    plt.plot(hist.history['accuracy'], '-o', label='treino')
    plt.plot(hist.history['val_accuracy'], '-o', label='teste')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.title('Perda')
    plt.plot(hist.history['loss'], '-o', label='treino')
    plt.plot(hist.history['val_loss'], '-o', label='teste')
    plt.legend(loc='lower right')
    
    plt.gcf().set_size_inches(15, 12)