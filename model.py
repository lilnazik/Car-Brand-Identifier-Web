import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import kagglehub


path = kagglehub.model_download("dekxrma/car-brand-identifier/keras/default")
print(path)
print(os.listdir(path))

model = load_model(model_path)


car_brand_labels = sorted(['bmw', 'ferrari', 'aston martin', 'mercedes-benz', 'rover', 'lamborghini', 'mini', 'cadillac', 'toyota',
                'volkswagen', 'ssangyong', 'nissan', 'alfa romeo', 'kia', 'citroen', 'bentley', 'dacia', 'ford', 'audi',
                'saab', 'infiniti', 'lotus', 'jaguar', 'smart', 'hundai', 'honda', 'chevrolet', 'tesla', 'rolls royce',
                'renault', 'seat', 'maserati', 'daihatsu', 'mg', 'volvo', 'dodge', 'fiat', 'vauxhall', 'land rover', 'jeep',
                'ds automobiles', 'peugeot', 'porsche', 'isuzu', 'suzuki', 'mazda', 'mitsubishi', 'chrysler', 'subaru',
                'skoda', 'mclaren', 'lexus', 'hyundai'])

IMG_SIZE = (300, 300)

def prepare_image(image: Image.Image, target_size=IMG_SIZE):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict(image: Image.Image):
    processed = prepare_image(image)
    preds = model.predict(processed)[0]

    top_idxs = preds.argsort()[-3:][::-1]

    results = []
    for idx in top_idxs:
        label = car_brand_labels[idx]
        prob = float(preds[idx])
        results.append((label, prob))

    return results
