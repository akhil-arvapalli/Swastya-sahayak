# # import streamlit as st
# # import google.generativeai as genai
# # import tensorflow as tf
# # import numpy as np
# # import cv2
# # from PyPDF2 import PdfReader  # For PDF report extraction
# # from fastapi import FastAPI, UploadFile, File, HTTPException
# # from fastapi.responses import JSONResponse
# # import uvicorn
# # import pytesseract
# # import re
# # import json

# # # Configure Gemini API
# # GEMINI_API_KEY =  # Replace with your API key
# # genai.configure(api_key=GEMINI_API_KEY)

# # # Load the trained model
# # MODEL_PATH = "D:/A2vp/final_model.keras"  # Update the path if needed
# # model = tf.keras.models.load_model(MODEL_PATH)

# # # Set dark theme
# # st.set_page_config(page_title="AI Health Assistant", layout="wide")

# # # Sidebar menu
# # st.sidebar.title("🔍 AI Health Assistant")
# # menu = st.sidebar.radio("Choose an Option", ["Chat with AI", "Upload Medical Images", "Upload Reports"])

# # # Image preprocessing function
# # def preprocess_input_image(image):
# #     img = cv2.resize(image, (224, 224))
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     img = img.astype(np.float32) / 255.0
# #     img = np.expand_dims(img, axis=0)
# #     return img

# # # PDF text extraction using PyPDF2
# # def extract_text_from_pdf(pdf_file):
# #     reader = PdfReader(pdf_file)
# #     text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
# #     return text if text else "No text found in the document."

# # # FastAPI setup
# # app = FastAPI(title="PDF Data Extraction API")

# # # OCR text extraction using Tesseract
# # def extract_text_with_ocr(image):
# #     text = pytesseract.image_to_string(image)
# #     return text

# # # Text parsing using RegEx
# # def parse_text(text):
# #     data = {}
# #     patterns = {
# #         "name": r"Name[:\s]*([A-Za-z\s]+)",
# #         "date": r"Date[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})",
# #         "amount": r"Amount[:\s]*\$?(\d+\.?\d*)",
# #         "id": r"ID[:\s]*(\w+-\d+)"
# #     }
# #     for key, pattern in patterns.items():
# #         match = re.search(pattern, text, re.IGNORECASE)
# #         data[key] = match.group(1) if match else "Not found"
# #     return data

# # @app.post("/extract-data/")
# # async def extract_data(pdf_file: UploadFile = File(...), file_format: str = "json"):
# #     if not pdf_file.filename.endswith(".pdf"):
# #         raise HTTPException(status_code=400, detail="Only PDF files are supported")
# #     try:
# #         text = extract_text_from_pdf(pdf_file.file)
# #         extracted_data = parse_text(text)
# #         return JSONResponse(content=extracted_data if file_format.lower() == "json" else {"error": "Unsupported format"})
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# # if menu == "Chat with AI":
# #     st.title("💬 AI Chatbot - Symptom Checker")
# #     user_input = st.text_input("Describe your symptoms:")
# #     if st.button("Get Diagnosis"):
# #         if user_input:
# #             try:
# #                 model = genai.GenerativeModel("gemini-1.5-pro")
# #                 prompt = ("You are an advanced medical AI agent trained to assist healthcare professionals. "
# #                           "Your role is to analyze symptoms and provide accurate, evidence-based insights. "
# #                           f"Given the symptoms: {user_input}, list possible medical conditions with brief explanations, "
# #                           "considering common and serious possibilities. Avoid giving definitive diagnoses; instead, "
# #                           "suggest conditions for further medical evaluation.")
# #                 response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=150, temperature=0.7))
# #                 st.write(response.text)
# #             except Exception as e:
# #                 st.error(f"An error occurred with the Gemini API: {e}")
# #         else:
# #             st.write("Please enter your symptoms.")

# # if menu == "Upload Medical Images":
# #     st.title("📸 Upload X-ray/MRI/CT for Diagnosis")
# #     uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
# #     if uploaded_file is not None:
# #         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
# #         image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
# #         st.image(image, caption="Uploaded Image", use_column_width=True)
# #         processed_image = preprocess_input_image(image)
# #         predictions = model.predict(processed_image)
# #         max_index = np.argmax(predictions)
# #         labels = ["Normal Lungs", "Viral Pneumonia", "Bacterial Pneumonia"]
# #         predicted_label = labels[max_index]
# #         st.write(f"🔍 **Model Prediction:** {predicted_label}")

# # if menu == "Upload Reports":
# #     st.title("📊 Upload Diagnostic Reports")
# #     uploaded_report = st.file_uploader("Upload PDF Report", type=["pdf"])
# #     if uploaded_report is not None:
# #         text = extract_text_from_pdf(uploaded_report)
# #         st.write(f"📑 Extracted Text: {text}")
# #         model = genai.GenerativeModel("gemini-1.5-pro")
# #         prompt = ("You are a medical expert. Analyze this diagnostic report. "
# #                   f"Report: {text}\nWhat insights can you provide?")
# #         response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=200, temperature=0.7))
# #         st.write(f"🔍 Analysis: {response.text}")

# # if __name__ == "__main__":
# #     uvicorn.run(app, host="0.0.0.0", port=8000)

# import streamlit as st
# import google.generativeai as genai
# import tensorflow as tf
# import numpy as np
# import cv2
# from PyPDF2 import PdfReader
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# import uvicorn
# import pytesseract
# import re
# import json
# import googlemaps

# # Configure Gemini API
# GEMINI_API_KEY = "YOUR_API_KEY_HERE"
# genai.configure(api_key=GEMINI_API_KEY)

# # Configure Google Maps API
# MAPS_API_KEY = "YOUR_API_KEY_HERE"
# gmaps = googlemaps.Client(key=MAPS_API_KEY)

# # Load the trained model
# MODEL_PATH = "D:/A2vp/final_model.keras"
# model = tf.keras.models.load_model(MODEL_PATH)

# # Set dark theme
# st.set_page_config(page_title="AI Health Assistant", layout="wide")

# # Sidebar menu
# st.sidebar.title("🔍 AI Health Assistant")
# menu = st.sidebar.radio("Choose an Option", ["Chat with AI", "Upload Medical Images", "Upload Reports", "Find Nearby Hospitals"])

# # Image preprocessing function
# def preprocess_input_image(image):
#     img = cv2.resize(image, (224, 224))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32) / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# # PDF text extraction using PyPDF2
# def extract_text_from_pdf(pdf_file):
#     reader = PdfReader(pdf_file)
#     text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
#     return text if text else "No text found in the document."

# # Find nearby hospitals
# def get_nearby_hospitals(lat=17.3902157, lng=78.3213376):
#     try:
#         places_result = gmaps.places_nearby(
#             location=(lat, lng),
#             radius=5000,
#             keyword='hospital'
#         )
#         hospital_list = []
#         if places_result.get('results'):
#             for place in places_result['results'][:5]:
#                 name = place['name']
#                 address = place.get('vicinity', 'N/A')
#                 rating = place.get('rating', 'N/A')
#                 map_url = f"https://www.google.com/maps/search/?api=1&query={place['geometry']['location']['lat']},{place['geometry']['location']['lng']}"
#                 hospital_list.append(f"🏥 {name}\n📍 Address: {address}\n⭐ Rating: {rating}\n🔗 [Open in Google Maps]({map_url})\n---")
#             st.markdown("### 🏥 Top Nearby Hospitals:")
#             for hospital in hospital_list:
#                 st.markdown(hospital)
#         else:
#             st.warning("No hospitals found in the area.")
#     except Exception as e:
#         st.error(f"Error finding hospitals: {str(e)}")

# if menu == "Chat with AI":
#     st.title("💬 AI Chatbot - Symptom Checker")
#     user_input = st.text_input("Describe your symptoms:")
#     if st.button("Get Diagnosis"):
#         if user_input:
#             try:
#                 model = genai.GenerativeModel("gemini-1.5-pro")
#                 prompt = ("You are an advanced medical AI agent trained to assist healthcare professionals. "
#                           "Your role is to analyze symptoms and provide accurate, evidence-based insights. "
#                           f"Given the symptoms: {user_input}, list possible medical conditions with brief explanations, "
#                           "considering common and serious possibilities. Avoid giving definitive diagnoses; instead, "
#                           "suggest conditions for further medical evaluation.")
#                 response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=150, temperature=0.7))
#                 st.write(response.text)
#             except Exception as e:
#                 st.error(f"An error occurred with the Gemini API: {e}")
#         else:
#             st.write("Please enter your symptoms.")

# if menu == "Upload Medical Images":
#     st.title("📸 Upload X-ray/MRI/CT for Diagnosis")
#     uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
#     if uploaded_file is not None:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         processed_image = preprocess_input_image(image)
#         predictions = model.predict(processed_image)
#         max_index = np.argmax(predictions)
#         labels = ["Normal Lungs", "Viral Pneumonia", "Bacterial Pneumonia"]
#         predicted_label = labels[max_index]
#         st.write(f"🔍 **Model Prediction:** {predicted_label}")

# if menu == "Upload Reports":
#     st.title("📊 Upload Diagnostic Reports")
#     uploaded_report = st.file_uploader("Upload PDF Report", type=["pdf"])
#     if uploaded_report is not None:
#         text = extract_text_from_pdf(uploaded_report)
#         st.write(f"📑 Extracted Text: {text}")
#         model = genai.GenerativeModel("gemini-1.5-pro")
#         prompt = ("You are a medical expert. Analyze this diagnostic report. "
#                   f"Report: {text}\nWhat insights can you provide?")
#         response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=200, temperature=0.7))
#         st.write(f"🔍 Analysis: {response.text}")

# if menu == "Find Nearby Hospitals":
#     st.title("🏥 Find Nearby Hospitals")
#     get_nearby_hospitals()

# if __name__ == "__main__":
#     st.write("AI Health Assistant is running...")

import streamlit as st
import google.generativeai as genai
import tensorflow as tf
import numpy as np
import cv2
from PyPDF2 import PdfReader
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pytesseract
import re
import json
import googlemaps

# Configure Gemini API
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=GEMINI_API_KEY)

# Configure Google Maps API
MAPS_API_KEY = "YOUR_API_KEY_HERE"
gmaps = googlemaps.Client(key=MAPS_API_KEY)

# Load the trained model
MODEL_PATH = "D:/A2vp/final_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Set dark theme
st.set_page_config(page_title="AI Health Assistant", layout="wide")

# Sidebar menu
st.sidebar.title("🔍 AI Health Assistant")
menu = st.sidebar.radio("Choose an Option", ["Chat with AI", "Upload Medical Images", "Upload Reports", "Find Nearby Hospitals"])

# Image preprocessing function
def preprocess_input_image(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# PDF text extraction using PyPDF2
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text if text else "No text found in the document."

# Find nearby hospitals
def get_nearby_hospitals(lat, lng):
    try:
        places_result = gmaps.places_nearby(
            location=(lat, lng),
            radius=5000,
            keyword='hospital'
        )
        return places_result.get('results', [])
    except Exception as e:
        st.error(f"Error finding hospitals: {str(e)}")
        return []

if menu == "Chat with AI":
    st.title("💬 AI Chatbot - Symptom Checker")
    user_input = st.text_input("Describe your symptoms:")
    if st.button("Get Diagnosis"):
        if user_input:
            try:
                model = genai.GenerativeModel("gemini-1.5-pro")
                prompt = ("You are an advanced medical AI agent trained to assist healthcare professionals. "
                          "Your role is to analyze symptoms and provide accurate, evidence-based insights. "
                          f"Given the symptoms: {user_input}, list possible medical conditions with brief explanations, "
                          "considering common and serious possibilities. Avoid giving definitive diagnoses; instead, "
                          "suggest conditions for further medical evaluation.")
                response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=150, temperature=0.7))
                st.write(response.text)
            except Exception as e:
                st.error(f"An error occurred with the Gemini API: {e}")
        else:
            st.write("Please enter your symptoms.")

if menu == "Upload Medical Images":
    st.title("📸 Upload X-ray/MRI/CT for Diagnosis")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_image = preprocess_input_image(image)
        predictions = model.predict(processed_image)
        max_index = np.argmax(predictions)
        labels = ["Normal Lungs", "Viral Pneumonia", "Bacterial Pneumonia"]
        predicted_label = labels[max_index]
        st.write(f"🔍 **Model Prediction:** {predicted_label}")

if menu == "Upload Reports":
    st.title("📊 Upload Diagnostic Reports")
    uploaded_report = st.file_uploader("Upload PDF Report", type=["pdf"])
    if uploaded_report is not None:
        text = extract_text_from_pdf(uploaded_report)
        st.write(f"📑 Extracted Text: {text}")
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = ("You are a medical expert. Analyze this diagnostic report. "
 
                  f"Report: {text}\nWhat insights can you provide?")
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=200, temperature=0.7))
        st.write(f"🔍 Analysis: {response.text}")

elif menu == "Find Nearby Hospitals":
    st.title("🏥 Find Nearby Hospitals")
    
    # Use default location
    DEFAULT_LAT, DEFAULT_LNG = 17.3902157, 78.3213376
    st.write(f"📌 Using your location: Latitude {DEFAULT_LAT}, Longitude {DEFAULT_LNG}")
    
    user_input = st.text_input("Enter any text to proceed:")
    if st.button("Find Hospitals") and user_input:
        hospitals = get_nearby_hospitals(DEFAULT_LAT, DEFAULT_LNG)
        if hospitals:
            st.success(f"Found {len(hospitals)} hospitals nearby!")
            for place in hospitals[:5]:
                with st.expander(f"🏥 {place['name']}"):
                    st.write(f"📍 Address: {place.get('vicinity', 'N/A')}")
                    st.write(f"⭐ Rating: {place.get('rating', 'N/A')}")
                    map_url = f"https://www.google.com/maps/search/?api=1&query={place['geometry']['location']['lat']},{place['geometry']['location']['lng']}"
                    st.markdown(f"[Open in Google Maps]({map_url})")
        else:
            st.warning("No hospitals found nearby.")

if __name__ == "__main__":
    st.write("")