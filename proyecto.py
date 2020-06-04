import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv3D
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os #interactuo con el sistema operativo
import cv2

dir=r"C:\Users\diegg\Downloads\clasificacionImg"
tipos = ["vidrio","metal","plastico","carton","basura","papel"]

"""
for tipo in tipos:
    path = os.path.join(dir,tipo) #une una o mas path, concatena varios paths de dir con cada etiqueta.
    class_num = tipos.index(tipo)

    for img in os.listdir(path):
        mat = cv2.imread(os.path.join(path,img),1)
        imaColor = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        arrIMA = cv2.resize(imaColor, (8, 8))
        #plt.imshow(imaColor)
        #plt.show()
        break
    break
azul, verde, rojo = cv2.split(arrIMA)
print(arrIMA)
#print(class_num)
#print(mat)
"""
#########################################
tamano = 256

#arr = cv2.resize(imaColor,(tamano,tamano))
#plt.imshow(arr)
#plt.show()

training = []
class_num = []

def createTraining():
    for tipo in tipos:
        path = os.path.join(dir, tipo)  # une una o mas path, concatena varios paths de dir con cada etiqueta.
        class_num = tipos.index(tipo) # asigna un numero a cada tipo
        print(class_num)
        #print(os.listdir(path))
        for img in os.listdir(path):
            try:
                mat = cv2.imread(os.path.join(path, img), 1)
                imaColor = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
                arr = cv2.resize(imaColor, (tamano, tamano))
                training.append([arr,class_num]) #dupla matriz imagen y numero de tipo
                #print(path)
                #plt.imshow(imaColor)
                #plt.show()
            except Exception as e:
               pass




createTraining()

#print(len(training))
#print(training.shape)

import random
random.shuffle(training)###############aclarar

X = []
Y = []
for f,l in training:
    X.append(f)
    Y.append(l)

X = np.array(X).reshape(-1,tamano,tamano,3)
Y = np.array(Y)
#print(X.shape)
X = X/255.0
#print(X[0])
#print(len(X))

#13:30 video
print("Voy pal modelo")

model = Sequential()

#model.add(Flatten(input_shape=(tamano,tamano,3)))
#model.add(Dense(64, activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(46, (3, 3), activation='relu'))
#model.add(MaxPooling2D())
model.add(Conv2D(32, (9, 9), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(6, activation='softmax'))
#model.add(Conv2D(96, (3, 3), activation='relu'))
#model.add(MaxPooling2D()

#model.add(Conv2D(16, (3, 3), activation='softmax'))
#model.add(MaxPooling2D(pool_size=(2, 2)))



#model.add(Dense(96, activation='relu'))



##model.add(Dense(16, activation='softmax'))



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, Y, batch_size=15, epochs=4)

"""
prueba = X[100]
prueba = (np.expand_dims(prueba,0))
predictions_single = model.predict(prueba)
print(predictions_single)
print(np.argmax(predictions_single[0]))
"""


#plt.imshow(X[100])
#plt.show()
#print(tf.__version__)





dirFotos=r"C:\Users\diegg\Downloads\pruebasClasificador3"
trainingFotos = []
for img in os.listdir(dirFotos):
    try:
        mat = cv2.imread(os.path.join(dirFotos, img), 1)
        imaColor = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        arr = cv2.resize(imaColor, (tamano, tamano))
        trainingFotos.append(arr)  # dupla matriz imagen y numero de tipo
        # print(path)
        # plt.imshow(imaColor)
        # plt.show()
    except Exception as e:
        pass

#X2 = []
#Y2 = []
#for f in trainingFotos:
#    X2.append(f)
#    Y2.append(l)

trainingFotos = np.array(trainingFotos).reshape(-1,tamano,tamano,3)
#Y2 = np.array(Y2)
#print(X2.shape)
X2 = trainingFotos/255.0

predicciones = model.predict(X2)
#print("Predicciones",predicciones)
def plot_image(i, predictions_array, img):
  predictions_array, img = predictions_array, img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)


  plt.xlabel("{} {:2.0f}% ".format(tipos[predicted_label], 100*np.max(predictions_array) ))


def plot_value_array(predictions_array):
  #predictions_array = predictions_array
  plt.grid(False)
  #plt.xticks(range(6))
  plt.xticks(range(len(tipos)), tipos, rotation=45)
  plt.yticks([])
  plt.bar(range(len(tipos)), predictions_array, color='black')
  plt.ylim([0, 1])



for i in range(0,len(trainingFotos)):
    plt.subplot(1,2,1)
    plot_image(i, predicciones[i], trainingFotos)
    plt.subplot(1,2,2)
    plot_value_array(predicciones[i])
    plt.show()





