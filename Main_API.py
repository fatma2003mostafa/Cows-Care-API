from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
from tensorflow.keras.models import load_model

app = FastAPI()

# تخزين آخر نتيجة
last_prediction = {
    "class": None,
    "confidence": None
}

# تحميل الموديل
MODEL_PATH = 'cattle_disease_model_mobilenetv2.h5'
model = load_model(MODEL_PATH)

# أسماء الأمراض حسب ترتيب طبقة الإخراج – عدلهم لو في ترتيب معين
class_names = ["Lumpy Skin Disease", "Cow Pox", "Healthy", "Ringworm", "FMD"]

# المعالجة المسبقة للصورة
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

# استجابة نجاح منظمة
def success_response(message: str, data: dict = None):
    return JSONResponse(content={
        "status": "success",
        "message": message,
        "data": data or {}
    })

# استجابة خطأ منظمة
def error_response(message: str, status_code: int = 400):
    return JSONResponse(content={
        "status": "error",
        "message": message,
        "data": {}
    }, status_code=status_code)

# نقطة فحص الحالة
@app.get("/health")
def health_check():
    try:
        model.summary()  # تأكد إن الموديل جاهز
        return success_response(message="API is healthy. Model is loaded and ready.")
    except:
        return error_response("Model failed to load.", status_code=500)

# نقطة معلومات عن الموديل
@app.get("/model_info")
def model_info():
    return success_response(
        message="Model information retrieved.",
        data={
            "model_file": MODEL_PATH,
            "number_of_classes": len(class_names),
            "class_names": class_names
        }
    )

@app.get("/")
def read_root():
    if last_prediction["class"] is not None:
        return success_response(
            message="Last prediction retrieved successfully.",
            data={
                "predicted_class": last_prediction["class"],
                "confidence_percentage": last_prediction["confidence"]
            }
        )
    else:
        return success_response(message="API is running. Ready to predict!")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global last_prediction
    try:
        contents = await file.read()

        if not contents:
            return error_response("No file uploaded or file is empty.", status_code=400)

        input_data = preprocess_image(contents)

        prediction = model.predict(input_data)
        predicted_index = int(np.argmax(prediction))
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        last_prediction["class"] = predicted_label
        last_prediction["confidence"] = round(confidence, 2)

        return success_response(
            message="Prediction successful.",
            data={
                "predicted_class": predicted_label,
                "confidence_percentage": round(confidence, 2),
                "all_confidences": {
                    class_names[i]: round(float(score) * 100, 2) for i, score in enumerate(prediction[0])
                }
            }
        )
    except HTTPException as e:
        return error_response(str(e.detail), status_code=e.status_code)
    except Exception as e:
        return error_response(f"Internal server error: {str(e)}", status_code=500)

@app.post("/reset")
def reset_prediction():
    global last_prediction
    last_prediction = {
        "class": None,
        "confidence": None
    }
    return success_response(message="Prediction has been reset successfully.")
