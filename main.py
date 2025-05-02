import streamlit as st
import tensorflow as tf
import numpy as np

# Set page config at the very top
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# Model prediction function
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")  # Updated model name
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert to batch format
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("🌱 PLANT DISEASE RECOGNITION SYSTEM")
    bg_image_path = r"C:\Users\ilaki\Desktop\Marutham Mozhi\homepage.webp"
    st.image(bg_image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant.
    2. **Analysis:** Our system processes the image using advanced deep learning techniques.
    3. **Results:** View the predicted disease and take action!

    ### Why Choose Us?
    - **Accurate:** Trained on a robust dataset of 87,000+ images.
    - **Simple UI:** Intuitive and user-friendly interface.
    - **Fast:** Get results in seconds!

    👉 Click on **Disease Recognition** in the sidebar to get started!
    """)

# About Page
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    #### 📊 Dataset Info
    This dataset is a refined version created using offline augmentation based on the original public dataset.

    - 87,000+ RGB images of healthy and diseased crop leaves
    - 38 disease categories
    - Divided as:
      - **Train:** 70,295 images
      - **Validation:** 17,572 images
      - **Test:** 33 images

    #### 🔗 Source
    [Original Dataset GitHub Repository](https://github.com/spMohanty/PlantVillage-Dataset)
    """)

# Disease Prediction Page
elif app_mode == "Disease Recognition":
    st.header("🩺 Disease Recognition")
    test_image = st.file_uploader("Upload a Plant Leaf Image", type=["jpg", "jpeg", "png", "webp"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("🔍 Analyzing the image...")

            try:
                result_index = model_prediction(test_image)

                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                st.success(f"✅ Our model predicts: **{class_name[result_index]}**")
            except Exception as e:
                st.error(f"Error loading model or making prediction.\n\n{e}")
