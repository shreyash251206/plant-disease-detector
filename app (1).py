import gradio as gr
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

print("Building model...")

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(38, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model ready!")

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

treatments = {
    'Apple___Apple_scab': 'Apple Scab: Dark lesions on leaves. Treatment: Remove infected leaves, apply fungicide, improve air circulation.',
    'Tomato___Late_blight': 'Late Blight: Large dark lesions. Treatment: Remove infected plants immediately, apply copper fungicide.',
    'Corn_(maize)___Common_rust_': 'Common Rust: Orange-brown pustules. Treatment: Apply fungicide if severe, remove infected leaves.',
    'Potato___Early_blight': 'Early Blight: Brown lesions with rings. Treatment: Apply fungicide, remove infected leaves.',
    'Grape___Black_rot': 'Black Rot: Tan spots, black berries. Treatment: Remove mummified fruit, apply fungicide.',
    'Tomato___Early_blight': 'Early Blight: Brown spots with rings. Treatment: Remove infected leaves, apply fungicide.',
    'Pepper,_bell___Bacterial_spot': 'Bacterial Spot: Dark spots with halos. Treatment: Remove plants, apply copper spray.',
}

def predict_disease(image):
    try:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx] * 100
        
        predicted_class = class_names[predicted_idx]
        disease_name = predicted_class.replace('___', ' - ').replace('_', ' ')
        
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        
        result = "# DIAGNOSIS REPORT\n\n"
        result += f"## {disease_name}\n\n"
        result += f"Confidence: {confidence:.1f}%\n\n"
        
        if confidence >= 60:
            result += "HIGH CONFIDENCE\n\n"
        elif confidence >= 30:
            result += "MODERATE CONFIDENCE\n\n"
        else:
            result += "LOW CONFIDENCE\n\n"
        
        result += "---\n\n"
        
        if predicted_class in treatments:
            result += f"## Treatment\n\n{treatments[predicted_class]}\n\n"
        elif 'healthy' in predicted_class.lower():
            plant = predicted_class.split('___')[0].replace('_', ' ').title()
            result += f"## Healthy Plant\n\nYour {plant} is healthy!\n\n"
        else:
            result += "## Treatment\n\nIsolate plant, remove infected parts, consult expert.\n\n"
        
        result += "---\n\n## Top 3 Possibilities\n\n"
        for i, idx in enumerate(top_3_idx, 1):
            name = class_names[idx].replace('___', ' - ').replace('_', ' ')
            conf = predictions[0][idx] * 100
            result += f"{i}. {name} ({conf:.1f}%)\n"
        
        result += "\n---\n\nDisclaimer: Educational tool. Consult experts for actual diagnosis."
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Upload Plant Leaf Image"),
    outputs=gr.Markdown(label="Diagnosis Report"),
    title="Plant Disease Detection System",
    description="Upload plant leaf image for disease diagnosis. Supports 38 diseases across 14 plants.",
    theme="soft"
)

demo.launch()