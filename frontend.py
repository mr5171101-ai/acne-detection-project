import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Face Acne Detection",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    # Ensure 'best.pt' is in the same directory or provide the full path
    return YOLO("best (1).pt")

model = load_model()

# -----------------------------
# UI Header
# -----------------------------
st.title("🩺 Face Acne Detection using YOLOv11")
st.markdown("Upload a face image to detect and analyze acne severity using your fine-tuned YOLO model.")

# -----------------------------
# File Uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(uploaded_file, use_container_width=True)

    if st.button("🔍 Detect Acne", use_container_width=True):
        with st.spinner("Analyzing image for acne spots..."):
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(
                bytearray(uploaded_file.read()),
                dtype=np.uint8
            )
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Perform YOLO prediction
            results = model.predict(
                image,
                conf=0.25
            )

            # Extract results
            res = results[0]
            detection_count = len(res.boxes)
            
            # Plot bounding boxes on image
            predicted = res.plot()
            predicted = cv2.cvtColor(
                predicted,
                cv2.COLOR_BGR2RGB
            )

            with col2:
                st.subheader("✅ Detection Result")
                st.image(
                    predicted,
                    use_container_width=True
                )

            # Display Analysis Summary
            st.success("Analysis Completed Successfully!")
            st.metric(label="Total Acne Spots Detected", value=detection_count)