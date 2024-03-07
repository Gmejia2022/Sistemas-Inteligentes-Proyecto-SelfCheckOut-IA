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
BATCH_SIZE = 256  # Define el número de muestras que se propagarán a través de la red antes de actualizar los pesos.
EPOCHS = 3  # Define cuántas veces el algoritmo de aprendizaje trabajará a través de todo el conjunto de datos.
image_size = 100  # Define el tamaño de las imágenes (100x100 píxeles) que se utilizarán para entrenar el modelo.
num_classes = 131  # Define el número total de clases distintas en el conjunto de datos de frutas.
print("----------------------------------------------")
print("Estos son los Hiperparámetros de evaluación:")
print("----------------------------------------------")
print("Batch_size:", BATCH_SIZE)
print("EPOCHS:", EPOCHS)
print("image_size:", image_size)
print("num_classes:", num_classes) 
print("----------------------------------------------")
# Pausa el programa y espera a que el usuario presione Enter
input("Presiona Enter para continuar...")
# Preparación del generador de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza los valores de los píxeles al rango [0, 1].
    validation_split=0.2  # Reserva el 20% de las imágenes para la validación.
)

# Carga y preparación del conjunto de datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    r'D:\WAM\ProyectoTensorGMA\archive\fruits-360_dataset\fruits-360\Training',  # Ruta al directorio de entrenamiento.
    target_size=(image_size, image_size),  # Redimensiona las imágenes a 100x100 píxeles.
    batch_size=BATCH_SIZE,
    class_mode='sparse',  # Utiliza etiquetas enteras para las clases.
    subset='training'  # Especifica que este es el conjunto de entrenamiento.
)
print("Carga de DataSet de Entrenamiento Satisfactoria...!")

# Imprimir el nombre de la clase y su índice correspondiente
for clase, indice in train_generator.class_indices.items():
    print(f"Clase: {clase}, Índice: {indice}")

# Lista de etiquetas de las clases
# Obtiene las etiquetas de las clases a partir de los nombres de las carpetas en el directorio de entrenamiento
etiquetas = list(train_generator.class_indices.keys())

# Carga y preparación del conjunto de datos de validación
validation_generator = train_datagen.flow_from_directory(
    r'D:\WAM\ProyectoTensorGMA\archive\fruits-360_dataset\fruits-360\Test',  # Ruta al directorio de pruebas.
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

print("Entrenando el Modelo; ha iniciado...!")

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

print("Evaluando el Modelo...!")
# Evaluación del modelo en el conjunto de datos de prueba
test_loss, test_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
print("Precisión en el conjunto de prueba:", test_accuracy)  # Muestra la precisión en el conjunto de prueba

# Visualización del rendimiento del modelo
mostrar_precisión_modelo(history)  # Función para visualizar la precisión del modelo a lo largo del entrenamiento


## Clasificacion y Prediccion sobre imagenes en el Disco
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
# Ruta a la imagen para clasificar
ruta_imagen = r'D:\WAM\ProyectoTensorGMA\Pruebas\banana-wam.jpg'  # La 'r' al principio indica una cadena de texto cruda para manejar las barras invertidas en la ruta

print("Esto es lo que evaluaremos:", ruta_imagen)
# Carga y redimensiona la imagen al tamaño esperado por el modelo (100x100 en este caso)
imagen = load_img(ruta_imagen, target_size=(100, 100))

# Convierte la imagen a un array de NumPy
imagen_array = img_to_array(imagen)

# Añade una dimensión extra al inicio para crear un lote de una sola imagen
imagen_tensor = np.expand_dims(imagen_array, axis=0)

# Normaliza el tensor de la imagen para que los valores de los píxeles estén en el rango [0, 1]
imagen_tensor /= 255.0

# Realiza la predicción con el modelo
y_hat = model.predict(imagen_tensor)

# Obtiene el índice de la clase con la mayor probabilidad
indice_predicho = np.argmax(y_hat[0])

# Obtiene el valor de probabilidad para la clase predicha
probabilidad_predicha = y_hat[0][indice_predicho] * 100  # Convertido a porcentaje

# Imprime la clase predicha y su probabilidad
print(f"Clase predicha: {etiquetas[indice_predicho]} con una probabilidad de {probabilidad_predicha:.2f}%")

print(f"Indice predicha: {indice_predicho} con una probabilidad de {probabilidad_predicha:.2f}%")

# Guarda el modelo completo en el formato SavedModel
# Ruta donde se guardará el modelo
ruta_modelo = r'D:\WAM\ProyectoTensorGMA\WAM_FruitModel\wam_FruitModel' 

# Guarda el modelo en la ruta especificada
model.save(ruta_modelo)

# Cargar el modelo desde la ruta especificada
#modelo_recargado = tf.keras.models.load_model(ruta_modelo)

