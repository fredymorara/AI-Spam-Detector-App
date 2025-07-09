import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pickle
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time

# --- Page and NLTK Setup ---
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="wide")

# Function to load Lottie animations from a URL
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Could not load animation: {e}")
        return None

# NLTK Resource Management (silent download)
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

def download_nltk_data():
    packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    for package in packages:
        try:
            # A more robust check for different package types
            resource_name = f"tokenizers/{package}" if package == "punkt" else f"corpora/{package}"
            nltk.data.find(resource_name)
        except LookupError:
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)

# Run the download function at app startup
download_nltk_data()


# --- Preprocessing and Model Loading (Cached for performance) ---
@st.cache_resource
def get_preprocessing_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words_set = set(stopwords.words('english'))
    return lemmatizer, stop_words_set

lemmatizer, stop_words_set = get_preprocessing_tools()

def improved_preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words_set]
    return " ".join(processed_tokens)

@st.cache_resource
def load_model():
    try:
        with open('spam_detector_svm_pipeline.pkl', 'rb') as file:
            model_pipeline = pickle.load(file)
        return model_pipeline
    except FileNotFoundError:
        st.error("Model file 'spam_detector_svm_pipeline.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model_pipeline = load_model()

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        border: 1px solid #4B8BBE;
        background-color: #4B8BBE;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: white;
        color: #4B8BBE;
        border: 1px solid #4B8BBE;
        transform: scale(1.02);
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- App Layout ---
col1, col2 = st.columns([2, 1.5], gap="large")

with col1:
    st.title("Spam Message Detector")
    st.markdown("Enter a message to see if our AI model classifies it as **Spam** or **Ham**.")
    
    user_input = st.text_area(
        "Message to check:", 
        height=250, 
        placeholder="E.g., 'Congratulations! You've won a free prize, click here...'"
    )
    
    if st.button("🔍 Analyze Message", type="primary"):
        if model_pipeline and user_input.strip():
            with st.spinner('Analyzing...'):
                time.sleep(1) # Simulate processing for better UX
                
                processed_input = improved_preprocess_text(user_input)
                prediction = model_pipeline.predict([processed_input])
                
                st.markdown("---")
                if prediction[0] == 1:
                    st.error("### 🚨 Result: This message is classified as SPAM.")
                else:
                    st.success("### ✅ Result: This message is classified as HAM (Legitimate).")
        elif not user_input.strip():
            st.warning("Please enter a message before analyzing.")
        else:
            st.error("Model not loaded. The application cannot make predictions.")

with col2:
    lottie_url = "https://lottie.host/80753066-e883-4a30-a212-914917a26c48/6r1H1Z68sS.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, speed=1, height=400, key="spam_detection_animation")

# --- Project Information Sidebar ---
st.sidebar.title("About this Project")
st.sidebar.info(
    "This AI-powered application classifies messages as spam or ham. "
    "It utilizes a Linear Support Vector Machine (SVM) model trained on the Kaggle SMS Spam Collection dataset."
)
st.sidebar.header("Group Members")
st.sidebar.markdown("""
- **Fredrick M. Morara** - *Group Leader*
- **Cleophas Kiama**
- **Trevor Maosa**
- **Morris Mwangi**
- **Noelah Amoni**
""")