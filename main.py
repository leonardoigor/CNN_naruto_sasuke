# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

import os

# Bibliotecas Auxiliares
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

shapeimg = 32
epochs = 100
path = 'img'
className = []
myList = os.listdir(path)
train_images = []
train_labels = []
test_images = []
test_labels = []
show_images = []
class_names = ['Naruto', 'Sasuke', 'Barba Branca']
for cl in myList:
    intoFolder = os.listdir(f"{path}/{cl}")
    for img in intoFolder:
        img = cv.imread(f"{path}/{cl}/{img}", 0)
        img = cv.resize(img, (shapeimg, shapeimg))
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        train_images.append(img)

        train_labels.append(int(cl))
train_images = np.array(train_images)
train_labels = np.array(train_labels)

for cl in myList:
    intoFolder = os.listdir(f"testimg/{cl}")
    for img in intoFolder:
        imgShow = cv.imread(f"testimg/{cl}/{img}")
        imgShow = cv.cvtColor(imgShow, cv.COLOR_RGB2BGR)
        imgShow = cv.resize(imgShow, (500, 500))
        show_images.append(imgShow)

        img = cv.imread(f"testimg/{cl}/{img}", 0)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        img = cv.resize(img, (shapeimg, shapeimg))
        test_images.append(img)
        test_labels.append(int(cl))
test_images = np.array(test_images)
test_labels = np.array(test_labels)


train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()


# model = keras.Sequential([

#     keras.layers.Flatten(input_shape=(shapeimg, shapeimg, 3)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(3, activation='softmax')
# ])
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
          input_shape=(shapeimg, shapeimg, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (2,2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.add(layers.Dense(3, activation='softmax')
          )

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=epochs)


predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    # plt.imshow(test_images[i], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    # print(predictions_array)
    # exit()
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(3), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

    # i = 4


for i in range(0, len(test_images)):
    print(i)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, show_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions,  test_labels)
    plt.show()


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
