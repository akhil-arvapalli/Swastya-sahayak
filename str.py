# # import streamlit as st
# # import openai

# # # Configure OpenAI API
# # openai.api_key = "YOUR_API_KEY_HERE"

# # # Set dark theme
# # st.set_page_config(page_title="AI Health Assistant", layout="wide")
# # st.markdown("""
# #     <style>
# #     body { background-color: #121212; color: white; }
# #     .stButton>button { background-color: #1E88E5; color: white; border-radius: 5px; }
# #     .stTextInput>div>div>input { color: white; }
# #     </style>
# #     """, unsafe_allow_html=True)

# # # ---- SIDEBAR MENU ----
# # st.sidebar.title("🔍 AI Health Assistant")
# # menu = st.sidebar.radio("Choose an Option", ["Chat with AI", "Upload Medical Images", "Upload Reports"])

# # # ---- CHAT INTERFACE ----
# # if menu == "Chat with AI":
# #     st.title("💬 AI Chatbot - Symptom Checker")
# #     user_input = st.text_input("Describe your symptoms:")

# #     if st.button("Get Diagnosis"):
# #         if user_input:
# #             try:
# #                 # Use OpenAI's GPT model for generating responses
# #                 response = openai.Completion.create(
# #                     engine="text-davinci-003",  # You can also use "text-davinci-002" or "text-davinci-001"
# #                     prompt=f"Based on symptoms: {user_input}, what are the possible medical conditions?",
# #                     temperature=0.7,
# #                     max_tokens=150,
# #                 )
# #                 st.write(response['choices'][0]['text']['strip'])
# #             except Exception as e:
# #                 st.error(f"An error occurred: {e}")
# #         else:
# #             st.write("Please enter your symptoms.")

# # # ---- UPLOAD MEDICAL IMAGES ----
# # elif menu == "Upload Medical Images":
# #     st.title("📸 Upload X-ray/MRI/CT for Diagnosis")
# #     uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# #     if uploaded_file is not None:
# #         image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
# #         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# #         st.image(image, caption="Uploaded Image", use_column_width=True)

# #         # Dummy CNN Prediction
# #         st.write("🔍 Model Prediction: Pneumonia Detected")  # Replace with actual CNN inference

# # # ---- UPLOAD REPORTS ----
# # elif menu == "Upload Reports":
# #     st.title("📊 Upload Diagnostic Reports")
# #     uploaded_report = st.file_uploader("Upload PDF or Text File", type=["pdf", "txt", "csv"])

# #     if uploaded_report is not None:
# #         st.write("📑 Extracted Data: [Dummy data here]")  # Placeholder, integrate OCR or ML here

# # # ---- FOOTER ----
# # st.sidebar.markdown("---")
# # st.sidebar.markdown("👨‍⚕ Built for Hackathon | Team AI")
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# import openai

# # Configure OpenAI API
# openai.api_key = "YOUR_API_KEY_HERE"

# # Set dark theme
# st.set_page_config(page_title="AI Health Assistant", layout="wide")
# st.markdown("""
#     <style>
#     body { background-color: #121212; color: white; }
#     .stButton>button { background-color: #1E88E5; color: white; border-radius: 5px; }
#     .stTextInput>div>div>input { color: white; }
#     </style>
#     """, unsafe_allow_html=True)

# # ---- SIDEBAR MENU ----
# st.sidebar.title("🔍 AI Health Assistant")
# menu = st.sidebar.radio("Choose an Option", ["Chat with AI", "Upload Medical Images", "Upload Reports"])

# # ---- CHAT INTERFACE ----
# if menu == "Chat with AI":
#     st.title("💬 AI Chatbot - Symptom Checker")
#     user_input = st.text_input("Describe your symptoms:")
    
#     if st.button("Get Diagnosis"):
#         if user_input:
#             try:
#                 # Use OpenAI's GPT-3.5-turbo model for generating responses
#                 response = openai.ChatCompletion.create(
#                     model="gpt-3.5-turbo",  # Use gpt-3.5-turbo
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant specialized in medical diagnosis."},
#                         {"role": "user", "content": f"Based on symptoms: {user_input}, what are the possible medical conditions?"},
#                     ],
#                     max_tokens=150,
#                     temperature=0.7,
#                 )
#                 st.write(response['choices'][0]['message']['content'])
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
#         else:
#             st.write("Please enter your symptoms.")

# # ---- UPLOAD MEDICAL IMAGES ----
# elif menu == "Upload Medical Images":
#     st.title("📸 Upload X-ray/MRI/CT for Diagnosis")
#     uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
#     if uploaded_file is not None:
#         image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         # Dummy CNN Prediction
#         st.write("🔍 Model Prediction: Pneumonia Detected")  # Replace with actual CNN inference

# # ---- UPLOAD REPORTS ----
# elif menu == "Upload Reports":
#     st.title("📊 Upload Diagnostic Reports")
#     uploaded_report = st.file_uploader("Upload PDF or Text File", type=["pdf", "txt", "csv"])
    
#     if uploaded_report is not None:
#         st.write("📑 Extracted Data: [Dummy data here]")  # Placeholder, integrate OCR or ML here

# # ---- FOOTER ----
# st.sidebar.markdown("---")
# st.sidebar.markdown("👨‍⚕ Built for Hackathon | Team AI")
import streamlit as st
import openai
import tensorflow as tf
import numpy as np
import cv2
from PyPDF2 import PdfReader  # For PDF report extraction

# Configure OpenAI API
openai.api_key = "sk-ijklmnopqrstuvwxijklmnopqrstuvwxijklmnop"  # Your provided OpenAI API key

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
model_choice = st.sidebar.selectbox("Select OpenAI Model", ["gpt-3.5-turbo", "gpt-4"])  # Common OpenAI models

# ---- CHAT INTERFACE ----
if menu == "Chat with AI":
    st.title("💬 AI Chatbot - Symptom Checker")
    user_input = st.text_input("Describe your symptoms:")
    
    if st.button("Get Diagnosis"):
        if user_input:
            try:
                # Use OpenAI's ChatCompletion API
                response = openai.ChatCompletion.create(
                    model=model_choice,  # Dynamic model selection from sidebar
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in medical diagnosis."},
                        {"role": "user", "content": f"Based on symptoms: {user_input}, what are the possible medical conditions?"}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                diagnosis = response['choices'][0]['message']['content']
                st.write(diagnosis)

            except Exception as e:
                st.error(f"An error occurred with the OpenAI API: {e}")
        else:
            st.write("Please enter your symptoms.")

# ---- UPLOAD MEDICAL IMAGES ----
elif menu == "Upload Medical Images":
    st.title("📸 Upload X-ray/MRI/CT for Diagnosis")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Dummy CNN Prediction (OpenAI can’t process images directly)
        st.write("🔍 Model Prediction: Pneumonia Detected")  # Replace with actual CNN inference

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

            # Analyze with OpenAI
            response = openai.ChatCompletion.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a medical expert. Analyze this diagnostic report."},
                    {"role": "user", "content": f"Report: {text}\nWhat insights can you provide?"}
                ],
                max_tokens=200,
                temperature=0.7
            )
            analysis = response['choices'][0]['message']['content']
            st.write(f"🔍 Analysis: {analysis}")

        except Exception as e:
            st.error(f"Error processing report: {e}")

# ---- FOOTER ----
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍⚕ Built for Hackathon | Team AI")