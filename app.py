import streamlit as st

# Set page configuration before any other Streamlit command
st.set_page_config(page_title="SkincareAI", page_icon="🧴")

import numpy as np
import cv2
from joblib import load
import joblib
import mahotas
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .header-title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .header-subtitle {
        text-align: center;
        font-size: 20px;
        margin-bottom: 30px;
        color: #34495e;
    }
    .section-title {
        color: #2c3e50;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.markdown("<div class='header-title'>🧴 AI-Based Skincare Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Get personalized skincare recommendations based on your skin analysis and lifestyle details.</div>", unsafe_allow_html=True)

# Load models
skin_model = joblib.load("models/skin_type_svm_model.pkl")
skin_pca = joblib.load("models/skin_type_pca.pkl")
skin_scaler = joblib.load("models/skin_type_scaler.pkl")

acne_model = joblib.load("models/svm_acne_vgg_model.pkl")

wrinkle_model = load("models/wrinkle_model.joblib")
wrinkle_pca = load("models/wrinkle_pca.joblib")
wrinkle_le = load("models/wrinkle_label_encoder.joblib")

skin_classes = ['dry', 'normal', 'oily']

# --- Feature extraction functions ---
def extract_skin_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_feat = np.concatenate([
        cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0,256,0,256,0,256]).flatten(),
        cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0,180,0,256,0,256]).flatten()
    ])

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)

    haralick_feat = mahotas.features.haralick(gray).mean(axis=0)
    return np.concatenate([color_feat, lbp_hist, haralick_feat])

def extract_acne_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # Load the VGG16 model (without top layers)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Extract features
    features = feature_extractor.predict(img)
    return features.flatten()

def extract_wrinkle_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick[:6]

# --- Prediction functions ---
def predict_skin_type(img):
    features = extract_skin_features(img)
    features_scaled = skin_scaler.transform([features])
    features_pca = skin_pca.transform(features_scaled)
    prediction = skin_model.predict(features_pca)[0]
    return skin_classes[prediction]

def predict_acne(img):
    features = extract_acne_features(img)
    pred = acne_model.predict([features])[0]
    prob = acne_model.predict_proba([features])[0][pred]
    if pred == 1 and prob >= 0.60:
        label = "Acne detected"
        return (label, prob * 100)
    elif pred == 1 and prob < 0.60:
        label = "Slight possibility of acne"
        return (label, prob * 100)
    else:
        label = "No acne"
        return (label, prob * 100) 

def predict_wrinkles(img):
    features = extract_wrinkle_features(img)
    features_pca = wrinkle_pca.transform([features])
    prediction = wrinkle_model.predict(features_pca)[0]
    prob = wrinkle_model.predict_proba(features_pca)[0]
    label = wrinkle_le.inverse_transform([prediction])[0]
    confidence = prob[prediction]
    if label.lower() == 'wrinkled' and confidence >= 0.98:
        final_label = 'wrinkled'
    elif label.lower() == 'wrinkled':
        final_label = 'not wrinkled'
    else:
        final_label = label.lower()
    return final_label

# --- Recommendation Logic ---
def generate_recommendation(skin_type, acne_lvl, wrink_lvl, age, profession, work_hours, free_time, using_products):
    acne_label, acne_prob = acne_lvl
    recommendations = []
    # Skin type recommendations
    if skin_type == 'dry':
        recommendations.extend([
            "💧 Dry skin: try a rich, hydrating moisturizer.",
            "🧴 Use gentle, non-stripping cleansers."
        ])
    elif skin_type == 'oily':
        recommendations.extend([
            "🌿 Oily skin: opt for oil-free cleansers and light moisturizers.",
            "💦 Consider salicylic acid to control shine."
        ])
    elif skin_type == 'normal':
        recommendations.extend([
            "👌 Normal skin: maintain balance with quality, simple products.",
            "☀️ Daily SPF is a must!"
        ])
    
    # Acne recommendations
    if acne_label == 'acne detected':
        if acne_prob > 80:
            recommendations.extend([
                "🚨 Severe acne: consult a dermatologist & use targeted treatments.",
                "🔍 Stick to non-comedogenic, gentle products."
            ])
        elif 60 <= acne_prob <= 80:
            recommendations.append("⚠️ Moderate acne: consider spot treatments & regular cleansing.")
        else:
            recommendations.append("🙂 Mild acne: maintain a clean routine & monitor your skin.")
    else:
        recommendations.append("👍 No acne signs: continue your good skincare habits.")
    
    # Wrinkle recommendations
    if wrink_lvl == "wrinkled":
        recommendations.extend([
            "⏳ Notice wrinkles: add retinol or peptide-based products at night.",
            "🌟 Use antioxidants to keep skin youthful."
        ])
    else:
        recommendations.append("😊 Smooth skin: a basic routine with SPF works well.")
    
    # Age-specific recommendations
    if age < 18:
        recommendations.append("🌱 Teen skin: use a gentle cleanser and light moisturizer.")
    elif 18 <= age < 30:
        recommendations.append("✨ Young skin: stick to SPF and occasional exfoliation.")
    elif 30 <= age < 45:
        recommendations.append("💡 Early aging: try anti-aging ingredients like peptides.")
    elif 45 <= age < 60:
        recommendations.append("🔆 Mature skin: focus on hydration and collagen-boosting products.")
    else:
        recommendations.append("🌟 Senior skin: prioritize nourishment and consider advanced care.")
    
    # Lifestyle recommendations based on profession and work schedule
    if profession in ["construction", "outdoor"]:
        recommendations.extend([
            "☀️ Outdoors: use SPF 50+ and reapply often.",
            "🚿 Wash off pollutants and sweat after work."
        ])
    elif profession in ["student", "jobless"]:
        recommendations.append("📚 Simple routine: consistency is key.")
    elif profession in ["office", "indoor"]:
        recommendations.extend([
            "🏢 Indoor work: keep skin moisturized and use a humidifier if needed.",
            "💻 Take screen breaks to reduce eye and facial strain."
        ])
    else:
        recommendations.append("🎯 Tailor your routine based on daily exposure.")
    
    if work_hours >= 10:
        recommendations.append("🌙 Long work hours: ensure a deep cleanse every night.")
    elif work_hours >= 4:
        recommendations.append("🕒 Regular hours: stay hydrated and always wear SPF.")
    
    # Free time recommendations
    if free_time < 1:
        recommendations.append("⏰ Little free time: stick to a quick 3-step routine (cleanse, moisturize, SPF).")
    elif 1 <= free_time < 3:
        recommendations.append("🕒 Some free time: enjoy a weekly face mask.")
    else:
        recommendations.append("🛀 Lots of free time: experiment with extra treatments or a spa day at home.")
    
    # Product usage suggestions
    if using_products == "no":
        recommendations.append("🆕 New to skincare? Start simple with cleanser, moisturizer, and SPF.")
    else:
        recommendations.append("🔍 Already using products? Check ingredients regularly to suit your skin's needs.")
    
    return recommendations

# --- Image Upload ---
uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

image_path = None  # Initialize image path

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_path = "temp.jpg"
    image.save(image_path)
    st.image(image_path, caption="Uploaded Image", use_container_width=True)

# --- User Details ---
st.markdown("<div class='section-title'>🧍 Lifestyle Details</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Enter your age", min_value=10, max_value=100, value=25)
    profession = st.selectbox("Enter your profession", options=["construction", "outdoor", "student", "jobless", "office", "indoor", "other"])
with col2:
    work_hours = st.number_input("Average work hours per day", min_value=0, max_value=24, value=6)
    free_time = st.number_input("Free time per day (hours)", min_value=0.0, max_value=24.0, value=2.0)
using_products = st.radio("Are you currently using skincare products?", options=["yes", "no"])

# --- Generate Recommendations ---
if st.button("💡 Generate Recommendations"):
    if image_path and os.path.exists(image_path):
        with st.spinner("Analyzing skin details..."):
            skin_type = predict_skin_type(image_path)
            acne_label, acne_prob = predict_acne(image_path)
            wrink_lvl = predict_wrinkles(image_path)
        st.markdown("<div class='section-title'>🔍 Analysis Results</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"Skin Type:\n{skin_type.capitalize()}")
        with col2:
            st.info(f"Acne Level:\n{acne_label.capitalize()}\n({acne_prob:.2f}%)")
        with col3:
            st.info(f"Wrinkle Level:\n{wrink_lvl.capitalize()}")

        st.markdown("<div class='section-title'>📋 Personalized Recommendations</div>", unsafe_allow_html=True)
        recs = generate_recommendation(
            skin_type.lower(),
            (acne_label.lower(), acne_prob),
            wrink_lvl.lower(),
            age, profession, work_hours, free_time,
            using_products.lower()
        )
        for i, rec in enumerate(recs, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.warning("Please upload a valid image before generating recommendations.")
