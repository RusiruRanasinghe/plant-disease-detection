import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

model = tf.keras.models.load_model("models/resnet50_best.h5")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}

def predict_disease(image):
    img = Image.fromarray(image.astype("uint8"), "RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    disease_name = class_names[predicted_class].replace("___", " - ").replace("_", " ")
    top5_idx = np.argsort(predictions[0])[-5:][::-1]
    top5 = {class_names[i].replace("___", " - ").replace("_", " "): float(predictions[0][i]) for i in top5_idx}
    result = f"Detected: {disease_name}\nConfidence: {confidence*100:.2f}%"
    if "healthy" in disease_name.lower():
        result += "\n✅ Plant appears healthy!"
    else:
        result += "\n⚠️ Disease detected. Consult an expert."
    return result, top5

demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(label="Upload Plant Leaf Image"),
    outputs=[gr.Textbox(label="Diagnosis"), gr.Label(label="Top 5 Predictions", num_top_classes=5)],
    title="Plant Disease Detection System",
    description="Upload a plant leaf image to detect diseases.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
