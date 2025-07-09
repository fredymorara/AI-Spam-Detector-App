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
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# NLTK Resource Management
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

def download_nltk_data():
    packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
    for package in packages:
        try:
            resource_name = f"tokenizers/{package}" if package in ['punkt', 'punkt_tab'] else f"corpora/{package}"
            nltk.data.find(resource_name)
        except LookupError:
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)

download_nltk_data()

# --- Preprocessing and Model Loading (Cached for performance) ---
@st.cache_resource
def get_preprocessing_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words_set = set(stopwords.words('english'))
    return lemmatizer, stop_words_set

lemmatizer, stop_words_set = get_preprocessing_tools()

def improved_preprocess_text(text):
    if not isinstance(text, str): text = str(text)
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
    except Exception:
        return None

model_pipeline = load_model()

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .stApp {
        background: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #4B8BBE;
        background-color: #4B8BBE;
        color: white;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: white;
        color: #4B8BBE;
        border: 1px solid #4B8BBE;
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# --- App Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Spam Email & SMS Detector")
    st.markdown("Enter a message to classify it as **Spam** or **Ham** (legitimate). Our AI model will analyze it instantly.")
    
    user_input = st.text_area("Enter the message to check:", height=200, placeholder="E.g., 'Congratulations! You've won a free prize...'")
    
    if st.button("🔍 Classify Message", type="primary"):
        if model_pipeline and user_input.strip():
            # Add a spinner for a better user experience
            with st.spinner('Analyzing your message...'):
                time.sleep(1) # Simulate processing time
                
                processed_input = improved_preprocess_text(user_input)
                prediction = model_pipeline.predict([processed_input])
                
                st.markdown("---")
                if prediction[0] == 1:
                    st.error("### 🚨 Prediction: This message is likely SPAM.")
                else:
                    st.success("### ✅ Prediction: This message is likely HAM (Legitimate).")
        elif not user_input.strip():
            st.warning("Please enter a message to classify.")
        else:
            st.error("Model not loaded. The application cannot make predictions.")

with col2:
    # Lottie animation
    lottie_url = "https://lottie.host/80753066-e883-4a30-a212-914917a26c48/6r1H1Z68sS.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, speed=1, height=300, key="initial")

# --- Project Information Sidebar ---
st.sidebar.title("About the Project")
st.sidebar.info(
    "This AI Mini-Project uses a Linear SVM model trained on the Kaggle SMS Spam Collection dataset to classify messages."
)
st.sidebar.header("Group Members")
st.sidebar.markdown("""
- **Fredrick M. Morara** - *Group Leader*
- **Cleophas Kiama**
- **Trevor Maosa**
- **Morris Mwangi**
- **Noelah Amoni**
""")