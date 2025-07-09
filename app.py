import streamlit as st
import pickle
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- NLTK Resource Management for Streamlit Cloud (Final Version) ---
# Define a directory for NLTK data within the app's file system
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add this new directory to NLTK's list of data paths
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Function to download NLTK data if it's not already present
def download_nltk_data():
    # THE FIX IS HERE: Added 'punkt_tab' to the list of packages.
    packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
    for package in packages:
        try:
            # A more robust check for different package types (corpora, tokenizers, etc.)
            resource_name = f"tokenizers/{package}" if package in ['punkt', 'punkt_tab'] else f"corpora/{package}"
            nltk.data.find(resource_name)
        except LookupError:
            st.info(f"Downloading NLTK package: {package}...")
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)
            st.info(f"'{package}' downloaded.")

# Run the download function at app startup
download_nltk_data()

# --- Preprocessing Function (MUST be identical to the one used for training) ---
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
    
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        st.error(f"Tokenization failed: {e}")
        return ""

    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words_set]
    return " ".join(processed_tokens)

# --- Load the Saved Model Pipeline ---
@st.cache_resource
def load_model():
    try:
        with open('spam_detector_svm_pipeline.pkl', 'rb') as file:
            model_pipeline = pickle.load(file)
        return model_pipeline
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'spam_detector_svm_pipeline.pkl' is in the GitHub repository.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model_pipeline = load_model()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Spam Detector", layout="wide")
    
st.title("📧 Spam Email & SMS Detector")
st.markdown("Enter a message below to classify it as either **Spam** or **Ham** (legitimate).")
st.markdown("---")

user_input = st.text_area("Enter the message to check:", height=150, placeholder="Type or paste your message here...")

if st.button("Classify Message", type="primary"):
    if model_pipeline is not None:
        if user_input.strip():
            processed_input = improved_preprocess_text(user_input)
            if processed_input:
                prediction = model_pipeline.predict([processed_input])
                st.markdown("---")
                if prediction[0] == 1:
                    st.error("### 🚨 Prediction: SPAM")
                else:
                    st.success("### ✅ Prediction: HAM (Legitimate)")
        else:
            st.warning("Please enter a message to classify.")
    else:
        st.error("The model could not be loaded. The application cannot make predictions.")

# --- Project Information Sidebar ---
st.sidebar.title("About the Project")
st.sidebar.info(
    "This is an AI Mini-Project designed to classify messages as spam or ham. "
    "It uses a Linear SVM model trained on the Kaggle SMS Spam Collection dataset."
)
st.sidebar.header("Group Members")
st.sidebar.markdown("""
- **Fredrick M. Morara** - *Group Leader*
- **Cleophas Kiama**
- **Trevor Maosa**
- **Morris Mwangi**
- **Noelah Amoni**
""")