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
    layout="centered"
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_classes():
    with open("classes.txt", "r", encoding="utf-8") as f:
        classes = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return classes


classes = load_classes()


# ============================================================
# TELUGU NAMES
# ============================================================

telugu_names = {
    "Bacterial leaf blight": "బాక్టీరియల్ లీఫ్ బ్లైట్",
    "Brown spot": "బ్రౌన్ స్పాట్",
    "Leaf smut": "లీఫ్ స్మట్"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "final_rice_disease_model.keras",
        compile=False
    )

    return model


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🌾 Rice Disease Detection")

st.write(
    "వరి ఆకు ఫోటోను అప్‌లోడ్ చేసి వ్యాధిని గుర్తించండి."
)

st.write(
    "Upload a rice leaf image to detect the possible disease."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write("Number of classes:", len(classes))
    st.write("Classes:", classes)
    st.write("Input size:", "224 × 224")


# ============================================================
# CAMERA
# ============================================================

camera_image = st.camera_input(
    "📷 Take a Photo / ఫోటో తీయండి"
)


# ============================================================
# GALLERY UPLOAD
# ============================================================

uploaded_image = st.file_uploader(
    "🖼️ Upload Photo / ఫోటో అప్‌లోడ్ చేయండి",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# SELECT IMAGE
# ============================================================

image_file = None

if camera_image is not None:

    image_file = camera_image

elif uploaded_image is not None:

    image_file = uploaded_image


# ============================================================
# IMAGE PREVIEW
# ============================================================

if image_file is not None:

    image = Image.open(image_file).convert("RGB")

    st.image(
        image,
        caption="Rice Leaf Image / వరి ఆకు ఫోటో",
        use_container_width=True
    )


    # ========================================================
    # DETECT BUTTON
    # ========================================================

    if st.button(
        "🔍 Detect Disease / వ్యాధిని గుర్తించండి",
        type="primary"
    ):

        with st.spinner(
            "Analyzing image... / ఫోటోను పరిశీలిస్తోంది..."
        ):

            # Resize to model input size
            image_resized = image.resize(
                (224, 224)
            )

            # Convert to NumPy
            image_array = np.array(
                image_resized
            ).astype(np.float32)

            # IMPORTANT:
            # Same preprocessing used during training
            image_array = preprocess_input(
                image_array
            )

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Prediction
            prediction = model.predict(
                image_array,
                verbose=0
            )

            probabilities = prediction[0]

            # Highest probability
            predicted_index = int(
                np.argmax(probabilities)
            )

            disease = classes[predicted_index]

            confidence = (
                float(probabilities[predicted_index])
                * 100
            )

            telugu_disease = telugu_names.get(
                disease,
                disease
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.success(
            "Prediction Completed / వ్యాధి గుర్తింపు పూర్తయింది"
        )

        st.subheader(
            "🌾 Result / ఫలితం"
        )


        # Disease
        st.write("### Disease / వ్యాధి")

        st.write(
            f"**{disease}**"
        )


        # Telugu name
        st.write("### తెలుగు పేరు")

        st.write(
            f"**{telugu_disease}**"
        )


        # Confidence
        st.write("### Confidence / నమ్మక స్థాయి")

        st.progress(
            min(confidence / 100, 1.0)
        )

        st.write(
            f"**{confidence:.2f}%**"
        )


        # ====================================================
        # CONFIDENCE MESSAGE
        # ====================================================

        if confidence >= 80:

            st.success(
                "High confidence / అధిక నమ్మక స్థాయి"
            )

        elif confidence >= 60:

            st.info(
                "Moderate confidence / మధ్యస్థ నమ్మక స్థాయి"
            )

        else:

            st.warning(
                "Low confidence / తక్కువ నమ్మక స్థాయి. "
                "Please upload a clear close-up image "
                "of the rice leaf."
            )


        # ====================================================
        # INFORMATION
        # ====================================================

        st.subheader(
            "🌱 Information / సమాచారం"
        )


        if disease == "Bacterial leaf blight":

            st.info(
                "English: The model detected possible "
                "Bacterial Leaf Blight.\n\n"
                "తెలుగు: మోడల్ బాక్టీరియల్ లీఫ్ బ్లైట్ "
                "ఉండే అవకాశం ఉందని గుర్తించింది."
            )


        elif disease == "Brown spot":

            st.info(
                "English: The model detected possible "
                "Brown Spot.\n\n"
                "తెలుగు: మోడల్ బ్రౌన్ స్పాట్ "
                "ఉండే అవకాశం ఉందని గుర్తించింది."
            )


        elif disease == "Leaf smut":

            st.info(
                "English: The model detected possible "
                "Leaf Smut.\n\n"
                "తెలుగు: మోడల్ లీఫ్ స్మట్ "
                "ఉండే అవకాశం ఉందని గుర్తించింది."
            )


        # ====================================================
        # ALL PROBABILITIES
        # ====================================================

        with st.expander(
            "📊 See all class probabilities"
        ):

            for i, class_name in enumerate(classes):

                probability = (
                    float(probabilities[i])
                    * 100
                )

                st.write(
                    f"**{class_name}: "
                    f"{probability:.2f}%**"
                )

                st.progress(
                    min(probability / 100, 1.0)
                )


        # ====================================================
        # NOTE
        # ====================================================

        st.caption(
            "Note: This prediction is based on the trained "
            "machine-learning model and image quality. "
            "It should not be considered a definitive "
            "agricultural diagnosis."
        )