from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from io import BytesIO
import gdown
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# إعداد النموذج
file_id = "1vwlsHu7rYkN1pi7AUphi4-M6iciVfWaG"  # ID من Google Drive
model_path = "down_detect_v1.keras"
download_url = f"https://drive.google.com/uc?id={file_id}"

# تحميل النموذج من Google Drive إذا لم يكن موجوداً
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(download_url, model_path, quiet=False)

# تحميل النموذج
model = load_model(model_path)
categories = ["Non-Standardexpected", "Standardnormal"]

# إعداد التطبيق
app = FastAPI()

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# معالجة الصورة
def preprocess_image(image_data):
    img = load_img(image_data, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# نقطة التنبؤ
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_data = BytesIO(contents)
        img_array = preprocess_image(image_data)

        prediction = model.predict(img_array, verbose=0)[0]
        predicted_class_index = int(np.argmax(prediction))
        predicted_class = categories[predicted_class_index]
        confidence = float(np.max(prediction) * 100)

        return JSONResponse(content={
            "result": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "predicted_class_index": predicted_class_index
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
