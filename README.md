# 🌿 Plant Disease Recognition System

A deep learning-based web application that identifies plant leaf diseases from uploaded images. Built using TensorFlow and Streamlit, this tool helps farmers, researchers, and agricultural professionals detect and respond to plant diseases early and effectively.

---

## 📘 About This Project

This project uses a Convolutional Neural Network (CNN) trained on a large dataset of over 87,000 images across 38 disease categories. Users can upload a plant image, and the app will predict the disease class.

---

## 🔍 Features

- 🌱 Identify 38 types of plant diseases
- 📷 Upload leaf images in `.jpg`, `.jpeg`, `.png`, or `.webp` format
- 🧠 Deep learning model using TensorFlow and Keras
- 💻 Simple and intuitive Streamlit UI
- 📊 Trained on a robust and diverse dataset

---

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **Backend/Model**: TensorFlow / Keras
- **Language**: Python
- **Deployment**: Localhost or Streamlit Cloud

---

## 🧠 Dataset Info

- Source: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- Images: 87,000+ RGB images
- Categories: 38 disease classes
- Format: RGB, 128x128 resized
- Augmented to increase data robustness

---

## 🧰 Installation & Usage

### 🔧 Setup Environment

1. Clone the repo:
    ```bash
    git clone https://github.com/yourusername/plant-disease-recognition.git
    cd plant-disease-recognition
    ```

2. (Optional) Create a virtual environment:
    ```bash
    conda create -n plant-env python=3.9
    conda activate plant-env
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the app:
    ```bash
    streamlit run main.py
    ```

---

## ⚙️ CUDA/GPU Setup (Optional)

To speed up training or predictions using GPU:

1. Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install [cuDNN](https://developer.nvidia.com/cudnn)
3. Verify TensorFlow detects your GPU:
    ```python
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    ```

Ensure your TensorFlow version matches your CUDA/cuDNN installation.

---

## 🗂️ Project Structure

plant-disease-recognition/
│
├── main.py # Streamlit frontend + prediction logic
├── trained_model.keras # Pretrained model
├── requirements.txt # Python dependencies
├── homepage.webp # Background image
└── README.md # This file


---

## 💡 How to Use

1. Navigate to `http://localhost:8501`
2. Use the sidebar to select:
   - **Home**: Welcome screen
   - **About**: Dataset & project info
   - **Disease Recognition**: Upload plant leaf and get prediction

---

## 📚 Topics

deep-learning streamlit tensorflow computer-vision agriculture plant-disease image-classification cnn keras

---

## 📌 To-Do / Improvements

- [ ] Add multi-language support
- [ ] Add mobile responsiveness
- [ ] Deploy to Streamlit Cloud or Hugging Face

---

## 🔗 Links

- [Streamlit](https://streamlit.io)
- [TensorFlow](https://tensorflow.org)
- [Dataset Source](https://github.com/spMohanty/PlantVillage-Dataset)

---
14-07-25
🌟 New: Gemini AI Integration
Marutham Mozhi now integrates Google Gemini API to provide natural language advice in Tamil for detected plant diseases.

After predicting the disease from the uploaded image, the app uses Gemini to:

✅ Explain the disease in simple farmer-friendly Tamil

🌿 Suggest natural treatments or best practices

🤖 Respond in real-time with relevant context

🔧 Gemini API Setup
To enable Gemini features in your local environment:

Get an API key from Google AI Studio
Create a .env file in your project root:
GEMINI_API_KEY=your_key_here
Make sure .env is added to your .gitignore

🧰 Required Python Packages

pip install -r requirements.txt
And ensure you’ve installed:

google-generativeai
python-dotenv
tensorflow
streamlit
Firebase Integration (Firestore)
To enable prediction logging and future insights, Firebase Firestore has been added:
Each prediction (disease label + timestamp) is saved in Firestore
Easily extendable for user tracking, dashboarding, or analytics
Requires a firebase_key.json service account for authentication

