import streamlit as st
import google.generativeai as genai
import numpy as np
import cv2
from PyPDF2 import PdfReader
import googlemaps

# Configure APIs
GEMINI_API_KEY = "AIzaSyC5UCYCNO9Y2riHbB4FPtgoaQvpKSJSVMw"
MAPS_API_KEY = "AIzaSyDZRvkzza8WKTu-AhlLKVoUtEwWedCywo8"
genai.configure(api_key=GEMINI_API_KEY)
gmaps = googlemaps.Client(key=MAPS_API_KEY)

# Default Location
DEFAULT_LAT, DEFAULT_LNG = 17.42, 78.39

def process_image(image):
    """Basic image processing without ML model"""
    try:
        mean_color = np.mean(image, axis=(0,1))
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        return {'mean_color': mean_color, 'brightness': brightness}
    except Exception as e:
        return str(e)

# Set page configuration
st.set_page_config(page_title="AI Health Assistant", layout="wide")

# Sidebar Menu
st.sidebar.title("🔍 AI Health Assistant")
menu = st.sidebar.radio("Choose an Option", ["Chat with AI", "Upload Medical Images", "Upload Reports", "Find Nearby Hospitals"])

if menu == "Chat with AI":
    st.title("💬 AI Chatbot - Symptom Checker")
    user_input = st.text_input("Describe your symptoms:")
    
    if st.button("Get Diagnosis") and user_input:
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            prompt = f"Given the symptoms: {user_input}, list possible medical conditions with brief explanations."
            response = model.generate_content(prompt)
            st.write(response.text)
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif menu == "Upload Medical Images":
    st.title("📸 Upload Medical Images")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Image Analysis: {process_image(image)}")

elif menu == "Upload Reports":
    st.title("📊 Upload Diagnostic Reports")
    uploaded_report = st.file_uploader("Upload PDF or Text File", type=["pdf", "txt", "csv"])
    
    if uploaded_report is not None:
        try:
            if uploaded_report.name.endswith(".pdf"):
                reader = PdfReader(uploaded_report)
                text = reader.pages[0].extract_text()
            else:
                text = uploaded_report.read().decode("utf-8")
            
            st.write(f"📑 Extracted Text: {text}")
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(f"Analyze this medical report: {text}")
            st.write(f"🔍 Analysis: {response.text}")
        except Exception as e:
            st.error(f"Error processing report: {e}")

elif menu == "Find Nearby Hospitals":
    st.title("🏥 Find Nearby Hospitals")
    
    # Use default location
    lat, lng = DEFAULT_LAT, DEFAULT_LNG
    st.write(f"📌 Using your location: Latitude {lat}, Longitude {lng}")
    

    user_input = st.text_input("Enter any text to proceed:")
    if st.button("Find Hospitals") and user_input:
            try:
                places_result = gmaps.places_nearby(
                    location=(lat, lng),
                    radius=5000,
                    keyword='hospital'
                )
                
                if places_result.get('results'):
                    st.success(f"Found {len(places_result['results'])} hospitals nearby!")
                    for place in places_result['results'][:5]:
                        with st.expander(f"🏥 {place['name']}"):
                            st.write(f"📍 Address: {place.get('vicinity', 'N/A')}")
                            st.write(f"⭐ Rating: {place.get('rating', 'N/A')}")
                            map_url = f"https://www.google.com/maps/search/?api=1&query={place['geometry']['location']['lat']},{place['geometry']['location']['lng']}"
                            st.markdown(f"[Open in Google Maps]({map_url})")
                else:
                    st.warning("No hospitals found nearby.")
            except Exception as e:
                st.error(f"Error finding hospitals: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍⚕ Built for Hackathon | Team AI")

