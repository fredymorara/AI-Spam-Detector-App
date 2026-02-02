import streamlit as st
import pickle
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time

# --- Page Configuration (set this first) ---
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📧",
    layout="centered"
)

# --- NLTK Resource Management for Streamlit Cloud ---
# This block defines a custom path and explicitly downloads all required packages.
@st.cache_resource
def setup_nltk():
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    # List of all packages needed, INCLUDING the problematic 'punkt_tab'
    packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
    for package in packages:
        try:
            # Check if the data is already in our custom path
            resource_name = f"tokenizers/{package}" if package in ['punkt', 'punkt_tab'] else f"corpora/{package}"
            nltk.data.find(resource_name)
        except LookupError:
            # If not found, download it to our custom directory
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)

# Run the setup function once at app startup
setup_nltk()


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
    
    try:
        tokens = word_tokenize(text) # This should now find all necessary data
    except Exception as e:
        st.error(f"Tokenization failed: {e}")
        return ""

    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words_set]
    return " ".join(processed_tokens)


@st.cache_resource
def load_model():
    try:
        with open('spam_detector_svm_pipeline.pkl', 'rb') as file:
            model_pipeline = pickle.load(file)
        return model_pipeline
    except Exception:
        st.error("Model file 'spam_detector_svm_pipeline.pkl' not found.")
        return None

model_pipeline = load_model()


# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        border: 2px solid #007bff;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: white;
        color: #007bff;
    }
</style>
""", unsafe_allow_html=True)


# --- App Interface ---
st.title("📧 Spam Message Detector")
st.markdown("Enter a message to see if our AI model classifies it as **Spam** or **Ham**.")

user_input = st.text_area(
    "Message to analyze:", 
    height=200, 
    placeholder="e.g., 'Congratulations you've won a prize, click here...'"
)

if st.button("Analyze Message"):
    if model_pipeline and user_input.strip():
        with st.spinner('Analyzing...'):
            time.sleep(1) 
            
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
        st.error("Model not loaded. Please check the application logs.")


# --- Project Information Sidebar ---
st.sidebar.title("About this Project")
st.sidebar.info(
    "This AI Mini-Project uses a Linear Support Vector Machine (SVM) model to classify messages. The model was trained on the Kaggle SMS Spam Collection dataset."
)
st.sidebar.header("Group Members")
st.sidebar.markdown("""
- **Fredrick M. Morara**
""")
