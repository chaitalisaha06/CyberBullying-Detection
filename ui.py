import streamlit as st
import os
from PIL import Image
from image_offensive_pipeline import process_single_image  # assuming your main pipeline is saved as `main_pipeline.py`

# ================= Streamlit UI =================

st.set_page_config(page_title="Image Offensive Content Detector", layout="wide")
st.title("🚫 Offensive Content Detector (Image-Based)")
st.markdown("Upload an image to analyze whether it contains offensive content based on pose, quality, and vision-language understanding.")

# Model paths (update if your files are in a different directory)
model_paths = {
    "pose_model": "D:\\Image Cyberbullying\\checkpoints\\pose\\sapiens_1b_coco_best_coco_AP_821_torchscript.pt2",
    "scaler": "scaler.pkl",
    "pca": "pca.pkl",
    "nn_model": "neural_network_trained_model1.h5"
}

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save uploaded file temporarily
    with open("temp_uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.read())
    
    with st.spinner("Processing image... This may take a moment ⏳"):
        try:
            result = process_single_image("temp_uploaded_image.jpg", model_paths)

            # Results display
            st.success("✅ Processing complete!")
            st.markdown(f"**Prediction:** `{result['prediction']}`")
            st.markdown(f"**Probability Offensive:** `{result['probability_offensive']}`")
            st.markdown(f"**Global Contrast Factor (GCF):** `{result['gcf']}`")
            st.markdown(f"**Global Color Consistency (GCC):** `{result['gcc']}`")

            with st.expander("📖 Vision-Language Model Description"):
                st.write(result["llava_description"])

            with st.expander("📝 Summary (FLAN-T5 Explanation)"):
                st.write(result["summary"])

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# Optional footer
st.markdown("---")
st.markdown("Made with ❤️ using PyTorch, TensorFlow, Transformers, and Streamlit")
