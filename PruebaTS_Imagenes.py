## Proyecto: UEES Sistemas Inteligentes
## Developer: GMEJIA
## Fecha: 04 de Marzo 2024
## Descripción: Prueba de Tensorflow
## Clasificar imágenes CIFAR-10
## Con Prediccion sobre Imagenes en el Disco

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
EPOCHS = 10  # Se debe de Probar con valores mayores

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)

elapsed_time = time.time() - start_time
print("Tiempo de entrenamiento:", elapsed_time, "segundos")

test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("Precisión en el conjunto de prueba:", test_accuracy)

mostrar_precisión_modelo(history)

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Ruta a la imagen
#ruta_imagen = r'C:\ML\Pruebas\Carro.jpg'  # La 'r' al principio indica una cadena de texto cruda para manejar las barras invertidas en la ruta
ruta_imagen = r'C:\ML\Pruebas\Barco.jpg'  # La 'r' al principio indica una cadena de texto cruda para manejar las barras invertidas en la ruta

# Carga y redimensiona la imagen a 32x32 píxeles
imagen = load_img(ruta_imagen, target_size=(32, 32))

# Convierte la imagen cargada a un array de NumPy
imagen_array = img_to_array(imagen)

# Añade una dimensión extra para convertirlo en un tensor de forma (1, 32, 32, 3)
imagen_tensor = np.expand_dims(imagen_array, axis=0)

# Normaliza el tensor para que los valores de los píxeles estén en el rango [0, 1]
imagen_tensor /= 255.0

# Realiza la predicción con el modelo
y_hat = model.predict(imagen_tensor)

# Obtiene el índice de la clase con la mayor probabilidad
indice_predicho = np.argmax(y_hat[0])

# Obtiene el valor de probabilidad para la clase predicha
probabilidad_predicha = y_hat[0][indice_predicho] * 100  # Convertido a porcentaje

# Imprime la clase predicha usando las etiquetas de CIFAR-10 y su probabilidad
print(f"Clase predicha: [{cifar10_labels[indice_predicho]}] con una probabilidad de {probabilidad_predicha:.2f}%")
