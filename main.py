import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gemini_helper import get_gemini_response

# 🔁 Cache model to speed up repeat predictions
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.keras")

def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home / முகப்பு", "About / பற்றி", "Disease Recognition / நோய் கண்டறிதல்"])

# Home Page
if app_mode == "Home / முகப்பு":
    st.header("🌿 Marutham Mozhi / மருதம் மொழி")
    bg_image_path = r"C:\Users\ilaki\Desktop\Marutham Mozhi\Homepage.png"
    st.image(bg_image_path, use_container_width=True)

    st.markdown("#### 🪔 *“வளரும் இலை; வளரும் வாழ்வு.”* — A leaf that grows, nurtures life.")

    st.markdown("""
    ## Welcome to **Marutham Mozhi** 🌾  
    **மருதம் மொழிக்கு வரவேற்கிறோம்!**

    _"Where the whisper of leaves tells ancient tales, and every green breath feeds the world."_  
    _"இலைகள் சொல்வது பண்டை கதைகள்; பசுமை மூச்சே உலகை உயிர்த்தக்காக்கிறது."_

    Plants are the silent givers — the root of our food, the soul of our ecosystems.  
    **மரங்கள் நம்மை போஷிக்கின்றன. ஆனால் தெரியாத நோய்கள் அவற்றைப் பாதிக்கின்றன.**

    **Marutham Mozhi** is a bridge — connecting traditional farming wisdom with modern AI.  
    **மருதம் மொழி** என்பது பாரம்பரிய வேளாண்மை அறிவையும், நவீன தொழில்நுட்பத்தையும் இணைக்கும் பாலமாகும்.

    👉 Start from the sidebar! / **வலப்பக்கம் பட்டியலில் தொடங்கு!**
    """)

    st.markdown("<hr><center>🌿 *Dedicated to the farmers, the land, and the language that feeds us.* 🌾</center>", unsafe_allow_html=True)

# About Page
elif app_mode == "About / பற்றி":
    st.header("📘 About the Project / திட்டத்தைப் பற்றி")
    st.markdown("""
    ## 📚 Dataset: The Roots of Knowledge  
    **தகவல் தொகுப்பு: அறிவின் வேர்கள்**

    We use the **PlantVillage Dataset** – with 87,000+ images from various crops and diseases.  
    **87,000+ படங்கள் கொண்ட தொகுப்பு, பல பயிர்கள் மற்றும் நோய்கள் அடங்கியுள்ளன.**

    - **Training / பயிற்சி:** 70,295
    - **Validation / சரிபார்ப்பு:** 17,572
    - **Testing / சோதனை:** 33

    *Marutham Mozhi is not just a tool — it's a tribute.*  
    **மருதம் மொழி என்பது கருவி மட்டுமல்ல; மரபுக்கான அஞ்சலி.**
    """)

# Prediction Page
elif app_mode == "Disease Recognition / நோய் கண்டறிதல்":
    st.header("🩺 Disease Recognition / தாவர நோய் கண்டறிதல்")
    test_image = st.file_uploader("📤 Upload a Plant Leaf Image / ஒரு இலைப் படத்தை பதிவேற்றவும்", type=["jpg", "jpeg", "png", "webp"])

    if test_image is not None:
        st.image(test_image, caption="📸 Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict / கணிப்பு"):
            st.snow()
            st.write("🧠 Analyzing the image... / படம் பகுப்பாய்வு செய்யப்படுகிறது...")

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

                disease_result = class_name[result_index]

                st.success(f"✅ Predicted: **{disease_result}**\n\n🎉 கணிக்கப்பட்ட நோய்: **{disease_result}**")

                # 🌱 Gemini API for treatment advice
                with st.spinner("🌿 Getting advice from Gemini..."):
                    advice = get_gemini_response(disease_result)

                if advice:
                    st.markdown("### 🤖 Gemini Says (in Tamil):")
                    st.write(advice)
                else:
                    st.warning("⚠️ Gemini could not return any advice.")

            except Exception as e:
                st.error(f"❌ Error during prediction or Gemini response: {e}")

    st.markdown("""<hr><center>🌿 *“உழுவார் உலகத்தார்க்கெல்லாம் எழுவார் முன்னனி தூணை.”* — *Thirukkural 1031*</center>""", unsafe_allow_html=True)
