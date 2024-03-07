## Proyecto: UEES Sistemas Inteligentes
## Developer: GMEJIA
## Fecha: 04 de Marzo 2024
## Descripción: Prueba de Tensorflow
## Carga el Modelo Entrenado y Realiza Predicciones

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Ruta donde está guardado tu modelo entrenado
ruta_modelo = r'C:\ML\GMA_FruitModel\gma_FruitModel'

# Carga el modelo guardado
model = tf.keras.models.load_model(ruta_modelo)

# Función para hacer una predicción con una imagen
def predecir_imagen(ruta_imagen, model, etiquetas):
    # Carga y preprocesa la imagen
    imagen = load_img(ruta_imagen, target_size=(100, 100))  # Asegúrate de que el tamaño sea el mismo que el utilizado durante el entrenamiento
    imagen_array = img_to_array(imagen)
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Crea un lote de una sola imagen
    imagen_array /= 255.0  # Normaliza la imagen

    # Hace la predicción
    predicciones = model.predict(imagen_array)
    indice_predicho = np.argmax(predicciones[0])
    #etiqueta_predicha = etiquetas[indice_predicho]

    # Imprime la predicción
    #print(f"Predicción: {etiqueta_predicha} con una confianza de {np.max(predicciones[0]) * 100:.2f}%")

    print(f"Indice predicha: {indice_predicho}  con una confianza de {np.max(predicciones[0]) * 100:.2f}%")

# Lista de etiquetas (debes reemplazar esto con tus propias etiquetas)
etiquetas = ['Apple', 'Banana', 'Orange', ...]  # Asegúrate de que las etiquetas coincidan con el orden de las clases utilizado durante el entrenamiento

# Ruta a la imagen que quieres clasificar
# Ruta a la imagen para clasificar
ruta_imagen = r'C:\ML\FrutasDataSet\Datos\fruits-360_dataset\fruits-360\test-multiple_fruits\cherries7.jpg'  # La 'r' al principio indica una cadena de texto cruda para manejar las barras invertidas en la ruta

# Hacer una predicción
predecir_imagen(ruta_imagen, model, etiquetas)
