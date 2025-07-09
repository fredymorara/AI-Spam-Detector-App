import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- NLTK Resource Setup ---
# These are downloaded once when the app starts up on the server
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4', quiet=True)
# Explicitly add the one that was causing issues
try:
    nltk.data.find('tokenizers/punkt_tab')
except nltk.downloader.DownloadError:
    nltk.download('punkt_tab', quiet=True)

# --- Preprocessing Function (MUST be identical to the one used for training) ---
lemmatizer = WordNetLemmatizer()
stop_words_set = set(stopwords.words('english'))

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

# --- Load the Saved Model Pipeline ---
# Use a function with a cache decorator to load the model only once
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