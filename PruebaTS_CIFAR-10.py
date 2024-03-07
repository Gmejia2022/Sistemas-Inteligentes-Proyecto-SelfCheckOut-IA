## Proyecto: UEES Sistemas Inteligentes
## Developer: GMEJIA
## Fecha: 04 de Marzo 2024
## Descripción: Prueba de Tensorflow
## Clasificar imágenes CIFAR-10

import datetime
import tensorflow as tf
import time

import numpy as np
import matplotlib.pyplot as plt

from Funciones.VisualizaPlot import mostrar_precisión_modelo

print("Version Tensor Flow ", tf.__version__)

# Carga el conjunto de datos CIFAR-10 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Carga de DataSet Satisfactoria...!")

# Definimos las etiquetas de texto para el conjunto de datos CIFAR-10
cifar10_labels = ["Avion", "Automovil", "Pajaro", "Gato", "Ciervo",
                  "Perro", "Rana", "Caballo", "Barco", "Camion"]

# Índice de la imagen, puedes elegir cualquier número entre 0 y 59,999
img_index = 5
# y_train contiene las etiquetas, que van del 0 al 9
label_index = y_train[img_index][0]
# Imprime la etiqueta, por ejemplo "2 Bird"
print("y = " + str(label_index) + " " + (cifar10_labels[label_index]))
# Muestra una de las imágenes del conjunto de datos de entrenamiento
#plt.imshow(x_train[img_index])  # Muestra la imagen seleccionada
#plt.show()  # Muestra la Imagen

# Las imágenes de CIFAR-10 ya están en el tamaño adecuado y tienen 3 canales de color.
# Normalizamos los datos dividiendo por 255.
x_train, x_test = x_train / 255.0, x_test / 255.0

print("X_train.shape: ", x_train.shape)

#Proceso para la Creacion de la Capas de la Red Neuronal
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

model = tf.keras.Sequential()

# Ajusta el 'input_shape' al formato de las imágenes CIFAR-10 (32x32 píxeles con 3 canales de color)
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# La capa final debe tener 10 unidades, correspondientes a las 10 clases de CIFAR-10
model.add(Dense(10, activation='softmax'))

model.summary()

# Configuración del entrenamiento
BATCH_SIZE = 1000
EPOCHS = 20  # Se debe de Probar con valores mayores

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)

elapsed_time = time.time() - start_time
print("Tiempo de entrenamiento:", elapsed_time, "segundos")

test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("Precisión en el conjunto de prueba:", test_accuracy)

mostrar_precisión_modelo(history)

y_hat = model.predict(x_test)

# Selecciona una muestra aleatoria de 15 índices del conjunto de pruebas
indices_aleatorios = np.random.choice(x_test.shape[0], size=15, replace=False)

print("Resultados de las predicciones:")
print("Índice\tPredicción\t\t\tReal")

# Itera sobre los índices seleccionados
for i, index in enumerate(indices_aleatorios):
    # Determina la etiqueta predicha (el índice con el valor máximo en el vector de predicción 'y_hat[index]')
    predict_index = np.argmax(y_hat[index])
    # Obtiene la etiqueta real de 'y_test'. Nota que 'y_test' es un array 2D, así que accedemos al primer elemento para obtener la etiqueta real
    true_index = y_test[index][0]
    # Imprime el índice de la imagen, la etiqueta predicha y la etiqueta real. Usa 'cifar10_labels' para obtener la etiqueta textual
    print(f"{index}\t{cifar10_labels[predict_index]}\t\t{cifar10_labels[true_index]}")
