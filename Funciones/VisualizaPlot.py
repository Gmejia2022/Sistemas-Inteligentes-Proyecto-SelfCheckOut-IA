
def mostrar_precisión_modelo(history):
    # Extrae la precisión de entrenamiento y validación de 'history'
    precisión_entrenamiento = history.history['accuracy']
    precisión_validación = history.history['val_accuracy']
    
    # Calcula el número de épocas
    épocas = range(1, len(precisión_entrenamiento) + 1)
    
    # Imprime la precisión de entrenamiento y validación por época
    print("Época\tPrecisión Entrenamiento\tPrecisión Validación")
    for época, acc_ent, acc_val in zip(épocas, precisión_entrenamiento, precisión_validación):
        print(f"{época}\t{acc_ent:.4f}\t\t\t{acc_val:.4f}")



