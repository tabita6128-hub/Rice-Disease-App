import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# PAGE SETTINGS
# ============================================================
st.set_page_config(
    page_title="Rice Disease Detection",
    page_icon="🌾",
    layout="wide"
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

# ============================================================
# LOAD CLASS NAMES & MODEL
# ============================================================
def load_classes():
    try:
        with open("classes.txt", "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes
    except:
        return ["Bacterial leaf blight", "Brown spot", "Leaf smut"]

classes = load_classes()

telugu_names = {
    "Bacterial leaf blight": "బాక్టీరియల్ లీఫ్ బ్లైట్",
    "Brown spot": "బ్రౌన్ స్పాట్",
    "Leaf smut": "లీఫ్ స్మట్"
}

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("final_rice_disease_model.keras", compile=False)
        return model
    except:
        return None

model = load_model()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/rice-bowl.png", width=70)
    st.title("Project Info")
    st.markdown("---")
    
    st.markdown("### 🏛️ College Name")
    st.write("**VSM College of Engineering**")
    st.write("(Autonomous)")
    
    st.markdown("### 👨‍🏫 Project Guide")
    st.write("**Mr. Abdul Aziz Md**")
    
    st.markdown("### 👥 Team Members")
    st.write("1. **K. Tabita**")
    st.write("2. **T. Deekshitha**")
    st.write("3. **N. Bindhu**")
    
    st.markdown("---")
    if st.button("🏠 Home Page కి వెళ్ళండి"):
        st.session_state.page = "home"
        st.session_state.show_camera = False
        st.rerun()

# ============================================================
# PAGE 1: HOME PAGE
# ============================================================
if st.session_state.page == "home":
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌾 Artificial Intelligence Career for Women (AICW)</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Rice Guard AI - Smart Disease Detection</h2>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style='text-align: center; max-width: 800px; margin: auto; font-size: 16px; color: #dddddd;'>
        An advanced AI-powered web application designed specifically for rice farmers and agricultural experts. 
        It leverages cutting-edge computer vision technology to instantly detect and identify common rice leaf diseases. 
        Once identified, the application provides instant management solutions, treatment recommendations, and preventive measures available in both Telugu and English. 
        Our goal is to empower farmers with quick, accurate, and actionable information to protect their crops and improve yields.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.write("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👋 స్వాగతం! వరి ఆకు వ్యాధులను ఏఐ (AI) ద్వారా గుర్తించడానికి క్రింది బటన్ పై క్లిక్ చేయండి.")
        
        lang = st.selectbox("Choose Language / భాషను ఎంచుకోండి:", ["English", "Telugu"])
        
        if st.button("Click Next ➡️", type="primary", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

# ============================================================
# PAGE 2: MAIN DETECTION & CHATBOT PAGE
# ============================================================
elif st.session_state.page == "main":
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.session_state.show_camera = False
        st.rerun()

    st.title("🌾 Rice Guard AI - Smart Disease Detection")
    st.write("వరి ఆకు ఫోటోను అప్‌లోడ్ చేసి వ్యాధిని గుర్తించండి మరియు చాట్‌బాట్ ద్వారా మరిన్ని వివరాలు తెలుసుకోండి.")

    with st.expander("Model Information"):
        st.write("Number of classes:", len(classes))
        st.write("Classes:", classes)
        st.write("Input size:", "224 × 224")

    camera_image = None
    
    # "Open Camera" బటన్ ఏర్పాటు
    if not st.session_state.show_camera:
        if st.button("📷 Open Camera / కెమెరాను తెರೆಯండి", type="secondary"):
            st.session_state.show_camera = True
            st.rerun()
    else:
        if st.button("❌ Close Camera / కెమెరాను మూసివేయండి"):
            st.session_state.show_camera = False
            st.rerun()
        
        camera_image = st.camera_input("📷 Take a Photo / ఫోటో తీయండి")

    uploaded_image = st.file_uploader("🖼️ Upload Photo / ఫోటో అప్‌లోడ్ చేయండి", type=["jpg", "jpeg", "png"])

    image_file = camera_image if camera_image is not None else uploaded_image

    if image_file is not None and "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = image_file.name
    elif image_file is not None and st.session_state.last_uploaded != image_file.name:
        st.session_state.prediction_done = False
        st.session_state.messages = []
        st.session_state.last_uploaded = image_file.name

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Rice Leaf Image / వరి ఆకు ఫోటో", use_container_width=True)

        if st.button("🔍 Detect Disease / వ్యాధిని గుర్తించండి", type="primary"):
            if model is None:
                st.error("Model file not found! Please check model path.")
            else:
                with st.spinner("Analyzing image... / ఫోటోను పరిశీలిస్తోంది..."):
                    image_resized = image.resize((224, 224))
                    image_array = np.array(image_resized).astype(np.float32)
                    image_array = preprocess_input(image_array)
                    image_array = np.expand_dims(image_array, axis=0)

                    prediction = model.predict(image_array, verbose=0)
                    probabilities = prediction[0]
                    predicted_index = int(np.argmax(probabilities))
                    
                    disease = classes[predicted_index]
                    confidence = float(probabilities[predicted_index]) * 100
                    telugu_disease = telugu_names.get(disease, disease)

                    st.session_state.prediction_done = True
                    st.session_state.disease = disease
                    st.session_state.telugu_disease = telugu_disease
                    st.session_state.confidence = confidence
                    st.session_state.probabilities = probabilities
                    st.session_state.messages = []
                    
                    try:
                        from google import genai
                        api_key = st.secrets["GEMINI_API_KEY"]
                        client = genai.Client(api_key=api_key)
                        
                        prompt = (
                            f"You are an expert agricultural AI assistant. The rice plant has been diagnosed with '{disease}' (Telugu: {telugu_disease}). "
                            f"Provide the response strictly separated into two sections: "
                            f"1. ENGLISH SECTION: Give detailed description, symptoms, and management/pesticides in English. "
                            f"2. TELUGU SECTION (తెలుగు విభాగం): Give the exact same detailed description, symptoms, and management/pesticides translated cleanly into Telugu. "
                            f"Do not mix them paragraph by paragraph. Keep them clearly separated."
                        )
                        response = client.models.generate_content(
                            model="gemini-3.6-flash", 
                            contents=prompt
                        )
                        st.session_state.ai_recommendation = response.text
                    except Exception as e:
                        st.session_state.ai_recommendation = f"Could not load AI recommendations: {e}"

    if st.session_state.prediction_done:
        disease = st.session_state.disease
        telugu_disease = st.session_state.telugu_disease
        confidence = st.session_state.confidence

        st.success("Prediction Completed / వ్యాధి గుర్తింపు పూర్తయింది")
        st.subheader("🌾 Result / ఫలితం")
        
        st.write("### Disease / వ్యాధి")
        st.write(f"**English:** {disease}  |  **Telugu:** {telugu_disease}")
        
        st.write(f"**Confidence / నమ్మక స్థాయి: {confidence:.2f}%**")
        st.progress(min(confidence / 100, 1.0))

        st.subheader("🤖 AI Recommendations (Side by Side) / సలహాలు (పక్కపక్కనే)")
        
        if hasattr(st.session_state, 'ai_recommendation') and st.session_state.ai_recommendation:
            text = st.session_state.ai_recommendation
            
            parts = text.split("TELUGU SECTION")
            eng_part = parts[0].replace("ENGLISH SECTION", "").strip()
            tel_part = parts[1].replace("(", "").replace(")", "").strip() if len(parts) > 1 else "తెలుగు సమాచారం అందుబాటులో ఉంది."

            col_eng, col_tel = st.columns(2)
            
            with col_eng:
                st.markdown("### 🇬🇧 English")
                st.markdown(eng_part)
                
            with col_tel:
                st.markdown("### 🇮🇳 తెలుగు")
                st.markdown(tel_part)

        st.write("---")
        st.subheader("💬 Ask more about this disease / ఈ వ్యాధి గురించి మరిన్ని ప్రశ్నలు అడగండి")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_question := st.chat_input("Type your question here... / మీ సందేహాన్ని ఇక్కడ టైప్ చేయండి..."):
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

            with st.chat_message("assistant"):
                with st.spinner("Thinking... / ఆలోచిస్తోంది..."):
                    try:
                        from google import genai
                        api_key = st.secrets["GEMINI_API_KEY"]
                        client = genai.Client(api_key=api_key)
                        
                        chat_prompt = (
                            f"You are a specialized agricultural assistant for a Rice Disease Detection system. "
                            f"The current plant is diagnosed with {disease} ({telugu_disease}). "
                            f"User question: {user_question}. "
                            f"Provide the response clearly split into English and Telugu sections side by side."
                        )
                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=chat_prompt,
                        )
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Error: {e}")