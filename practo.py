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
from apify_client import ApifyClient

# Configure Gemini API
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=GEMINI_API_KEY)

# Configure Google Maps API
MAPS_API_KEY = "YOUR_API_KEY_HERE"
gmaps = googlemaps.Client(key=MAPS_API_KEY)

# Configure Apify API Token
APIFY_API_TOKEN = "YOUR_API_KEY_HERE"
apify_client = ApifyClient(APIFY_API_TOKEN)

# Load the trained model
MODEL_PATH = "D:/A2vp/final_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Set dark theme
st.set_page_config(page_title="AI Health Assistant", layout="wide")

# Sidebar menu
st.sidebar.title("🔍 AI Health Assistant")
menu = st.sidebar.radio("Choose an Option", ["Chat with AI", "Upload Medical Images", "Upload Reports", "Find Nearby Hospitals", "Find Doctors"])

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

# Find nearby hospitals using Google Maps API
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

# Find doctors using Apify Practo Doctor Scraper API
def get_doctors(speciality, city, max_items=3):
    try:
        run_input = {
            "searchUrls": [f"https://www.practo.com/search/doctors?city={city}&q={speciality}"],
            "maxItems": max_items,
        }
        run = apify_client.actor("easyapi/practo-doctor-scraper").call(run_input=run_input)
        dataset_id = run["defaultDatasetId"]
        doctors = list(apify_client.dataset(dataset_id).iterate_items())
        return doctors[:max_items]
    except Exception as e:
        st.error(f"Error fetching doctor data: {str(e)}")
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
        st.write(f"🔍 Model Prediction: {predicted_label}")

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
    # st.write(f"📌 Using your location: Latitude {DEFAULT_LAT}, Longitude {DEFAULT_LNG}")
    
    user_input = st.text_input("Enter any text to proceed:")
    if st.button("Find Hospitals") and user_input:
        hospitals = get_nearby_hospitals(DEFAULT_LAT, DEFAULT_LNG)
        if hospitals:
            st.success(f"Found {len(hospitals)} hospitals nearby!")
            for place in hospitals[:len(hospitals)]:
                with st.expander(f"🏥 {place['name']}"):
                    st.write(f"📍 Address: {place.get('vicinity', 'N/A')}")
                    st.write(f"⭐ Rating: {place.get('rating', 'N/A')}")
                    map_url = f"https://www.google.com/maps/search/?api=1&query={place['geometry']['location']['lat']},{place['geometry']['location']['lng']}"
                    st.markdown(f"[Open in Google Maps]({map_url})")
        else:
            st.warning("No hospitals found nearby.")

elif menu == "Find Doctors":
    st.title("👨‍⚕ Find Top Doctors")
    
    speciality = st.text_input("Enter Speciality (e.g., Pulmonologist):")
    city = st.text_input("Enter City (e.g., Gandipet):")
    
    if speciality and city and st.button("Search Doctors"):
        doctors = get_doctors(speciality=speciality, city=city)
        
        if doctors:
            st.success(f"Top {len(doctors)} {speciality}s in {city}:")
            for doctor in doctors:
                with st.expander(f"{doctor['name']} ({doctor['specialization']})"):
                    st.write(f"📍 Address: {doctor.get('clinic_address', 'N/A')}")
                    st.write(f"⭐ Rating: {doctor.get('rating', 'N/A')}")
                    fee_info = doctor.get('fees', 'N/A')
                    fee_str = f"{fee_info}" if fee_info != 'N/A' else 'Not Available'
                    st.write(f"💵 Fees: {fee_str}")
        else:
            st.warning(f"No top-rated {speciality}s found in {city}.")