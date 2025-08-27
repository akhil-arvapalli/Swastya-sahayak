import streamlit as st
import google.generativeai as genai
import tensorflow as tf
import numpy as np
import cv2
from PyPDF2 import PdfReader  # For PDF report extraction

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBUuEMlPfisln_gEs5szqdQe5ANhPZ9-oU"  # Use your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Load the trained model
MODEL_PATH = "D:/A2vp/final_model.keras"  # Update the path if needed
model = tf.keras.models.load_model(MODEL_PATH)

# Set dark theme
st.set_page_config(page_title="AI Health Assistant", layout="wide")
st.markdown("""
    <style>
    body { background-color: #121212; color: white; }
    .stButton>button { background-color: #1E88E5; color: white; border-radius: 5px; }
    .stTextInput>div>div>input { color: white; }
    </style>
    """, unsafe_allow_html=True)

# ---- SIDEBAR MENU ----
st.sidebar.title("🔍 AI Health Assistant")
menu = st.sidebar.radio("Choose an Option", ["Chat with AI", "Upload Medical Images", "Upload Reports"])
model_choice = st.sidebar.selectbox("Select Gemini Model", ["gemini-1.5-pro", "gemini-1.0-pro"])  # Gemini models

# ---- IMAGE PREPROCESSING FUNCTION ----
def preprocess_input_image(image):
    # Resize the image to the target size (224x224)
    img = cv2.resize(image, (224, 224))

    # Convert BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize the image (scale pixel values to range [0, 1])
    img = img.astype(np.float32) / 255.0

    # Reshape the image to add batch dimension (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

# ---- CHAT INTERFACE ----
if menu == "Chat with AI":
    st.title("💬 AI Chatbot - Symptom Checker")
    user_input = st.text_input("Describe your symptoms:")
    
    if st.button("Get Diagnosis"):
        if user_input:
            try:
                # Initialize the Gemini model
                model = genai.GenerativeModel(model_choice)
                
                # Prepare the prompt
                prompt = (
    "You are an advanced medical AI agent trained to assist healthcare professionals. "
    "Your role is to analyze symptoms and provide accurate, evidence-based insights. "
    f"Given the symptoms: {user_input}, list possible medical conditions with brief explanations, "
    "considering common and serious possibilities. Avoid giving definitive diagnoses; instead, "
    "suggest conditions for further medical evaluation."
)
                
                # Generate response
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=150,
                        temperature=0.7
                    )
                )
                diagnosis = response.text
                st.write(diagnosis)

            except Exception as e:
                st.error(f"An error occurred with the Gemini API: {e}")
        else:
            st.write("Please enter your symptoms.")

# ---- UPLOAD MEDICAL IMAGES ----
elif menu == "Upload Medical Images":
    st.title("📸 Upload X-ray/MRI/CT for Diagnosis")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert file to OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_input_image(image)

        # Predict using the model
        predictions = model.predict(processed_image)
        max_index = np.argmax(predictions)

        # Map index to corresponding label
        labels = ["Normal Lungs", "Viral Pneumonia", "Bacterial Pneumonia"]
        predicted_label = labels[max_index]

        # Display prediction
        st.write(f"🔍 **Model Prediction:** {predicted_label}")

# ---- UPLOAD REPORTS ----
elif menu == "Upload Reports":
    st.title("📊 Upload Diagnostic Reports")
    uploaded_report = st.file_uploader("Upload PDF or Text File", type=["pdf", "txt", "csv"])
    
    if uploaded_report is not None:
        try:
            # Extract text based on file type
            if uploaded_report.name.endswith(".pdf"):
                reader = PdfReader(uploaded_report)
                text = reader.pages[0].extract_text()  # First page (extend for multi-page if needed)
            elif uploaded_report.name.endswith(".txt"):
                text = uploaded_report.read().decode("utf-8")
            elif uploaded_report.name.endswith(".csv"):
                text = uploaded_report.read().decode("utf-8")  # Basic handling
            else:
                text = "Unsupported file format"

            st.write(f"📑 Extracted Text: {text}")

            # Analyze with Gemini
            model = genai.GenerativeModel(model_choice)
            prompt = (
                "You are a medical expert. Analyze this diagnostic report. "
                f"Report: {text}\nWhat insights can you provide?"
            )
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.7
                )
            )
            analysis = response.text
            st.write(f"🔍 Analysis: {analysis}")

        except Exception as e:
            st.error(f"Error processing report: {e}")

# ---- FOOTER ----
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍⚕ Built for Hackathon | Team AI")
