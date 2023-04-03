from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory('petsImages/', 
                                          target_size = (200, 200),
                                          batch_size = 10,
                                          class_mode = 'binary')

validation_dataset = train.flow_from_directory('validation/', 
                                          target_size = (200, 200),
                                          batch_size = 10,
                                          class_mode = 'binary')

print(train_dataset.class_indices)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape = (200,200,3)),
    tf.keras.layers.MaxPool2D(2,2), 
    #
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2), 
    #
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2), 
    #
    tf.keras.layers.Flatten(),
    ##
    tf.keras.layers.Dense(512, activation='relu'),
    ##
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(learning_rate=0.001), metrics=['accuracy'])

#configuring the training
model_fit = model.fit(train_dataset, validation_data= validation_dataset, batch_size=32, epochs=10)


dir_path = './test'
for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '//' + i, target_size = (200, 200))
    plt.imshow(img)

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis= 0)
    images = np.vstack([X])
    val = model.predict(images)

    print('--------------------------------')
    print('Image name: ' + i + '. Value: ' + str(val))

    if val == 0:
        print('I found a kitty!')
    elif val == 1:
        print('I found a dog!')
    else:
        print('Pet not found')
        
    print('--------------------------------')