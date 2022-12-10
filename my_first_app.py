# импорт необходимых библиотек 
import io 
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0, EfficientNetV2B0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# Декоратор указывает, что данные будут загружаться только один раз и кэшироваться для будущего использования.
@st.cache(allow_output_mutation=True)
# Функция для загрузки модели машинного обучения из библиотеки
def load_model():
    return EfficientNetB0(weights='imagenet')

# Функция для загрузки модели машинного обучения из библиотеки
def load_modelV2():
    return EfficientNetV2B0(weights='imagenet')

# Функцая обработки изображения
def preprocess_image(img):
    img = img.resize((224, 224)) # изменение размера изображения
    x = image.img_to_array(img) # преобразование изображения в массив numpy
    x = np.expand_dims(x, axis=0) # добавление новой оси в массив
    x = preprocess_input(x) # стандартная предварительная обработка (та же, что применялась при обучении)
    return x

# Функция загрузки изображения
def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания') 
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

# Функция вывода результатов работы
def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], cl[2])


model = load_model() # загрузка модели
modelV2 = load_modelV2() # загрузка модели


st.title('Классификации изображений в облаке Streamlit') # вывод шапки
img = load_image() # загрузка изображения
result = st.button('Распознать изображение') # присвоение статуса по нажатию кнопки

if result:
    x = preprocess_image(img) # предобработка загруженного изображения
    preds = model.predict(x) # работа модели машинного обучения
    preds2 = modelV2.predict(x) # работа модели машинного обучения
    st.write('**Результаты распознавания по модели EfficientNetB0:**') 
    print_predictions(preds) # вывод результатов
    st.write('**Результаты распознавания по модели EfficientNetV2B0:**') 
    print_predictions(preds2) # вывод результатов
