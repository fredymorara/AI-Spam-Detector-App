# AI Spam Detector Web Application

This repository contains the source code for a live Spam/Ham classification app deployed on Streamlit Community Cloud.

---

## 🚀 Live Application

You can access and interact with the live application here:

**➡️ [https://aispamdetector.streamlit.app/](https://aispamdetector.streamlit.app/)**

---

## Overview

This application uses a **Linear Support Vector Machine (SVM)** model to classify text messages as either "Spam" or "Ham" (legitimate). It was developed as part of an AI mini-project.

### Features
-   **Text Input:** A simple interface to enter any email or SMS message.
-   **Instant Classification:** Get a real-time prediction from the trained AI model.
-   **Built with Streamlit:** A fast and easy-to-use framework for building data apps.

## About the Project

This deployment repository contains only the files necessary to run the web application. The complete project, including the data analysis, model training, and evaluation notebook, can be found in our main project repository.

**➡️ [View the Full Project on GitHub](https://github.com/fredymorara/ai-spam-detector-mini-project)**

## How to Run Locally

If you wish to run this app on your local machine:

1.  **Prerequisites:** Python 3.8+ installed.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/fredymorara/AI-Spam-Detector-App.git
    cd AI-Spam-Detector-App
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The application will open in your default web browser.