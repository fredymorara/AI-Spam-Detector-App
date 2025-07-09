import streamlit as st
import pickle
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- NLTK Resource Management for Streamlit Cloud (Direct Loading) ---
# Define a directory for NLTK data within the app's file system
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")

# Add this new directory to NLTK's list of data paths
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Manually load the 'punkt' tokenizer to avoid LookupError
# This is a robust way to ensure it's available.
try:
    punkt_path = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    if not os.path.exists(punkt_path):
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    
    # Load stopwords and wordnet data which are less problematic
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
    
except Exception as e:
    st.error(f"Failed to setup NLTK resources: {e}")

# --- Preprocessing Function (MUST be identical to the one used for training) ---
@st.cache_resource
def get_preprocessing_tools():
    lemmatizer = WordNetLemmatizer()
    # Load stopwords from the downloaded files
    try:
        stop_words_set = set(stopwords.words('english'))
    except LookupError:
        st.error("Stopwords not found. Please ensure the 'stopwords' NLTK data is in the repository.")
        stop_words_set = set() # Fallback to empty set
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
    
    # Use the loaded tokenizer
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        st.error(f"Tokenization failed: {e}")
        return "" # Return empty string on failure

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
            # Preprocess the input
            processed_input = improved_preprocess_text(user_input)
            
            # Make prediction
            if processed_input: # Only predict if preprocessing was successful
                prediction = model_pipeline.predict([processed_input])
                
                # Display the result
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