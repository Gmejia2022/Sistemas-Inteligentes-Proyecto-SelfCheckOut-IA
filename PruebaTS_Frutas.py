## Proyecto: UEES Sistemas Inteligentes
## Developer: GMEJIA
## Fecha: 04 de Marzo 2024
## Descripción: Prueba de Tensorflow
## Clasificar imágenes con DataSet de Frutas

# Importaciones necesarias
import datetime
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Importación de funciones adicionales para la visualización
from Funciones.VisualizaPlot import mostrar_precisión_modelo

# Configuración del entrenamiento
BATCH_SIZE = 32  # Define el número de muestras que se propagarán a través de la red antes de actualizar los pesos.
EPOCHS = 1  # Define cuántas veces el algoritmo de aprendizaje trabajará a través de todo el conjunto de datos.
image_size = 100  # Define el tamaño de las imágenes (100x100 píxeles) que se utilizarán para entrenar el modelo.
num_classes = 131  # Define el número total de clases distintas en el conjunto de datos de frutas.

# Preparación del generador de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza los valores de los píxeles al rango [0, 1].
    validation_split=0.2  # Reserva el 20% de las imágenes para la validación.
)

# Carga y preparación del conjunto de datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    r'C:\ML\FrutasDataSet\Datos\fruits-360_dataset\fruits-360\Training',  # Ruta al directorio de entrenamiento.
    target_size=(image_size, image_size),  # Redimensiona las imágenes a 100x100 píxeles.
    batch_size=BATCH_SIZE,
    class_mode='sparse',  # Utiliza etiquetas enteras para las clases.
    subset='training'  # Especifica que este es el conjunto de entrenamiento.
)
print("Carga de DataSet de Entrenamiento Satisfactoria...!")

# Carga y preparación del conjunto de datos de validación
validation_generator = train_datagen.flow_from_directory(
    r'C:\ML\FrutasDataSet\Datos\fruits-360_dataset\fruits-360\Test',  # Ruta al directorio de pruebas.
    target_size=(image_size, image_size),  # Redimensiona las imágenes a 100x100 píxeles.
    batch_size=BATCH_SIZE,
    class_mode='sparse',  # Utiliza etiquetas enteras para las clases.
)
print("Carga de DataSet de Validacion Satisfactoria...!")

# Creación de la red neuronal
model = tf.keras.Sequential([
    # Capas convolucionales y de pooling para la extracción de características
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Capa de aplanado para convertir la matriz 3D en un vector 1D
    tf.keras.layers.Flatten(),

    # Capas densas para la clasificación
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Capa de salida con activación softmax para clasificación multiclase
])

print("Creacion del Modelo Satisfactoria...!")
# Compilación del modelo
model.compile(optimizer='adam',  # Optimizador Adam para el ajuste de los pesos
              loss='sparse_categorical_crossentropy',  # Función de pérdida para clases enteras
              metrics=['accuracy'])  # Métrica para evaluar el rendimiento del modelo

print("Compilacion del Modelo Satisfactoria...!")
# Entrenamiento del modelo
start_time = time.time()  # Registro del tiempo de inicio

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,  # Define el número de pasos por época
    epochs=EPOCHS,  # Número de épocas para entrenar el modelo
    validation_data=validation_generator,  # Datos para la validación
    validation_steps=validation_generator.samples // BATCH_SIZE,  # Define el número de pasos de validación
    verbose=1  # Muestra el progreso del entrenamiento
)
print("Entrenamiento del Modelo Satisfactoria...!")
elapsed_time = time.time() - start_time  # Cálculo del tiempo total de entrenamiento
print("Tiempo de entrenamiento:", elapsed_time, "segundos")

# Evaluación del modelo en el conjunto de datos de prueba
test_loss, test_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
print("Precisión en el conjunto de prueba:", test_accuracy)  # Muestra la precisión en el conjunto de prueba

# Visualización del rendimiento del modelo
mostrar_precisión_modelo(history)  # Función para visualizar la precisión del modelo a lo largo del entrenamiento

