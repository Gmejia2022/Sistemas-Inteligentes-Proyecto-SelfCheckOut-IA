## Proyecto: UEES Sistemas Inteligentes
## Developer: GMEJIA
## Fecha: 04 de Marzo 2024
## Descripción: Prueba de Tensorflow
## Classify Fashion-MNIST

import datetime
import tensorflow as tf
import time

import numpy as np
import matplotlib.pyplot as plt

from Funciones.VisualizaPlot import mostrar_precisión_modelo

print("Version Tensor Flow ", tf.__version__)

# ' el Data Set se Baja en: %USERPROFILE%\.keras\datasets\fashion-mnist
# Esta función de la API de TensorFlow que facilita la descarga y carga del conjunto de datos Fashion MNIST directamente desde sus servidores.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("Carga de DataSet Satisfactoria...!")

# Definimos las etiquetas de texto para el conjunto de datos Fashion MNIST
fashion_mnist_labels = ["T-shirt/top",  # índice 0
                        "Trouser",      # índice 1
                        "Pullover",     # índice 2
                        "Dress",        # índice 3
                        "Coat",         # índice 4
                        "Sandal",       # índice 5
                        "Shirt",        # índice 6
                        "Sneaker",      # índice 7
                        "Bag",          # índice 8
                        "Ankle boot"]   # índice 9

# Índice de la imagen, puedes elegir cualquier número entre 0 y 59,999
img_index = 5
# y_train contiene las etiquetas, que van del 0 al 9
label_index = y_train[img_index]
# Imprime la etiqueta, por ejemplo "2 Pullover"
print("y = " + str(label_index) + " " + (fashion_mnist_labels[label_index]))
# Muestra una de las imágenes del conjunto de datos de entrenamiento
#plt.imshow(x_train[img_index])  # Muestra la imagen seleccionada
#plt.show()  # Muestra la Imagen

# Definimos las dimensiones de ancho y alto para las imágenes, en este caso 28x28 píxeles
w, h = 28, 28

# Redimensionamos 'x_train' para adaptarlo a un formato adecuado para un modelo de aprendizaje profundo,
# cambiando su forma a (número de imágenes, ancho, alto, canales de color).
# El '1' al final indica que las imágenes son en escala de grises (un solo canal de color).
x_train = x_train.reshape(x_train.shape[0], w, h, 1)

# Realizamos la misma operación de redimensionamiento para 'x_test', 
# asegurando que tenga la misma forma que 'x_train' para la coherencia en la evaluación del modelo.
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# Consultamos la nueva forma de 'x_train' para verificar el redimensionamiento.
# Esto debería devolver una tupla indicando el número total de imágenes, 
# el ancho, el alto y la cantidad de canales de color.
x_train.shape

print("X_train.shape: ",x_train.shape)

#Proceso para la Creacion de la Capas de la Red Neuronal
# Importamos las capas necesarias desde Keras para construir el modelo
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

# Inicializamos el modelo secuencial, que nos permite añadir capas una tras otra
model = tf.keras.Sequential()

# Añadimos la primera capa convolucional con 64 filtros, un tamaño de kernel de 2x2,
# relleno 'same' para mantener las dimensiones de la imagen, activación ReLU y definimos
# el tamaño de entrada para la primera capa, que son imágenes de 28x28 píxeles con un canal (escala de grises)
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))

# Añadimos una capa de MaxPooling para reducir la dimensionalidad espacial de los mapas de características,
# utilizando un tamaño de pool de 2x2
model.add(MaxPooling2D(pool_size=2))

# Añadimos una capa de Dropout para reducir el overfitting durante el entrenamiento,
# descartando el 30% de las conexiones entre las capas
model.add(Dropout(0.3))

# Añadimos otra capa convolucional, esta vez con 32 filtros y las mismas especificaciones
# de tamaño de kernel y relleno que la primera capa convolucional
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))

# Repetimos la capa de MaxPooling y Dropout para seguir reduciendo la dimensionalidad
# y prevenir el overfitting
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

# Aplanamos los mapas de características para convertirlos en un vector único,
# lo que permite pasar de la representación 2D a una representación 1D
model.add(Flatten())

# Añadimos una capa densamente conectada (o completamente conectada) con 256 unidades
# y función de activación ReLU
model.add(Dense(256, activation='relu'))

# Añadimos otra capa de Dropout para reducir aún más el overfitting
model.add(Dropout(0.5))

# Finalmente, añadimos una capa densa con 10 unidades, correspondientes a las 10 clases
# posibles de salida, utilizando la función de activación softmax para obtener
# probabilidades de clase
model.add(Dense(10, activation='softmax'))

# Llamamos al método summary() para imprimir un resumen del modelo, mostrando todas
# las capas, su tipo, forma de salida y número de parámetros
model.summary()

# Definimos el tamaño del lote para el entrenamiento. Cada lote tendrá 1000 ejemplos.
BATCH_SIZE = 1000

# Definimos el número de EPOCHS para el entrenamiento. El modelo se entrenará durante 20 iteraciones completas del conjunto de datos.
#EPOCHS = 20
EPOCHS = 1

# Compilamos el modelo, configurando su función de pérdida, el optimizador y la métrica que queremos monitorizar.
# La 'sparse_categorical_crossentropy' es adecuada para clasificación multiclase cuando las etiquetas son enteros.
# 'adam' es un optimizador popular que ajusta automáticamente la tasa de aprendizaje durante el entrenamiento.
# 'accuracy' indica que queremos monitorizar la precisión del modelo durante el entrenamiento y la validación.
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenamos el modelo en el conjunto de datos. Utilizamos un 'magic command' de Jupyter (%time) para medir el tiempo que tarda el entrenamiento.
# 'x_train' y 'y_train' son los datos de entrenamiento y sus etiquetas respectivamente.
# 'epochs=EPOCHS' indica el número total de épocas para entrenar el modelo.
# 'batch_size=BATCH_SIZE' establece el tamaño del lote para el entrenamiento.
# 'validation_split=0.2' reserva el 20% de los datos de entrenamiento para la validación, ayudando a monitorizar el overfitting.
# 'verbose=1' activa los logs detallados durante el entrenamiento, mostrando el progreso de cada época.

# Comienza a medir el tiempo
start_time = time.time()

# Entrena el modelo
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)

# Calcula el tiempo transcurrido
elapsed_time = time.time() - start_time

# Imprime el tiempo transcurrido
print("Tiempo de entrenamiento:", elapsed_time, "segundos")

# Evalúa el modelo en el conjunto de datos de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

# Imprime la precisión del modelo en el conjunto de datos de prueba
print("Precisión en el conjunto de prueba:", test_accuracy)

# Mostramos la Precisión del Modelo
mostrar_precisión_modelo(history)

# Realiza predicciones en el conjunto de datos de prueba 'x_test' usando el modelo entrenado
y_hat = model.predict(x_test)

save_path = "C:\ML\Imagenes"
filename = f"frame_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
full_path = f"{save_path}/{filename}"

# Selecciona una muestra aleatoria de 15 índices del conjunto de pruebas
indices_aleatorios = np.random.choice(x_test.shape[0], size=15, replace=False)

print("Resultados de las predicciones:")
print("Índice\tPredicción\t\tReal")

# Itera sobre los índices seleccionados
for i, index in enumerate(indices_aleatorios):
    # Determina la etiqueta predicha (el índice con el valor máximo en el vector de predicción 'y_hat[index]')
    predict_index = np.argmax(y_hat[index])
    # Obtiene la etiqueta real de 'y_test'
    true_index = y_test[index]
    # Imprime el índice de la imagen, la etiqueta predicha y la etiqueta real
    print(f"{index}\t{fashion_mnist_labels[predict_index]}\t{fashion_mnist_labels[true_index]}")

# Guarda el Modelo para ser Utilizado Posterirmente
model.save("C:\ML\Modelo")

