
from google.colab import drive
drive.mount('/content/drive')
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

"""### Paso 2: Preparar el Dataset"""

# Directorio donde están las imágenes
train_dir = "/content/drive/My Drive/PLAGAS"

# Crear un generador de datos con aumento de datos
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

"""### Paso 3: Crear y Entrenar el Modelo"""

# Cargar el modelo base preentrenado
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

# Congelar las primeras capas del modelo base
base_model.trainable = True
# Elegir cuántas capas descongelar
fine_tune_at = 100  # Puedes ajustar este número
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))

# Crear el modelo completo
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Añadir Dropout
    tf.keras.layers.BatchNormalization(),  # Añadir Batch Normalization
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compilar el modelo con una tasa de aprendizaje baja
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(train_generator,
          validation_data=validation_generator,
          epochs=50,
          callbacks=[early_stopping, lr_schedule])

"""### Paso 3.1"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# Después de entrenar el modelo procedemos a generar la matriz de confusión
# Generar predicciones
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Generar y mostrar la Matriz de Confusión
cm = confusion_matrix(y_true, y_pred_classes)

# Mostrar la matriz de confusión con etiquetas y una barra de color
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Generar y mostrar el informe de clasificación
report = classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys())
print("Classification Report:\n", report)
print('')
print('CURVA ROC')
print('')
# Calcular y mostrar la Curva ROC
n_classes = len(validation_generator.class_indices)
y_true_bin = label_binarize(y_true, classes=range(n_classes))
y_pred_bin = y_pred

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'{list(validation_generator.class_indices.keys())[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos (False Positive Rate)')
plt.ylabel('Tasa de Verdaderos Positivos (True Positive Rate)')
plt.title('Curva ROC para Clasificación de Plagas en Cultivo de Brócoli')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

"""### Paso 4: Cargar el Modelo y Hacer Predicciones"""

# Función para predecir el tipo de plaga
def predict_image(img_path, threshold=0.7):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar la imagen

    # Hacer la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)  # Obtener la confianza más alta

    # Mapear la clase predicha al nombre de la plaga
    class_names = list(train_generator.class_indices.keys())
    predicted_label = class_names[predicted_class[0]]

    # Comprobar si la confianza es menor que el umbral
    if confidence < threshold:
        return "Resultado no encontrado"

    # Verificar si la predicción es errónea comparando la confianza con las otras clases
    sorted_predictions = np.sort(predictions[0])[::-1]  # Ordenar predicciones en orden descendente
    second_best_confidence = sorted_predictions[1]  # Obtener la segunda mejor predicción

    # Si la diferencia entre la mejor predicción y la segunda mejor es pequeña, evitar un falso positivo
    if confidence - second_best_confidence < 0.2:
        return "Resultado no encontrado"

    return predicted_label

import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import uvicorn
import threading

app = FastAPI()

#Url para saber si la api esta activa

@app.get("/")
def read_root():
    return {"message": "¡Bienvenido a la API REST con FastAPI desde el movil v4"}

#Metodo para recibir la imagen
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Leer el archivo subido
        contents = await file.read()

        # Convertir los bytes en una imagen utilizando PIL
        img = Image.open(BytesIO(contents))

        # Guardar la imagen en el sistema de archivos
        img.save("decoded_image.png")

        # Suponiendo que tienes una función para predecir basada en la imagen
        prediccion = predict_image("decoded_image.png")

        # Mostrar la imagen usando matplotlib
        plt.imshow(img)
        plt.axis('off')  # Opcional: para ocultar los ejes
        plt.show()

        # return {"message": f"La imagen pertenece a la plaga: {prediccion}"}
        return {"message": f"{prediccion}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Ejecutar Uvicorn en segundo plano en el puerto 8000
def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run).start()

cloudflared tunnel --url http://localhost:8000
