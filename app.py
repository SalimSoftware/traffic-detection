import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# 1. Page Config (makes it look like an app on mobile)
st.set_page_config(page_title="Moroccan Sign Detector", layout="centered")

# 2. Load Model (Cached so it doesn't reload every time you click)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure 'best.pt' is in the folder.")

st.title("🇲🇦 Traffic Sign Detector")
st.write("Point your camera at a road sign.")

# 3. Input: Camera (Works on Android/iOS browsers)
img_buffer = st.camera_input("Scan Road")

if img_buffer is not None:
    # Convert buffer to PIL Image
    image = Image.open(img_buffer)

    # 4. Run YOLO
    results = model(image)

    # 5. Draw Boxes & Show Image
    # 'plot()' draws the bounding boxes on the image for us
    res_plotted = results[0].plot()
    
    # Display the processed image
    st.image(res_plotted, use_container_width=True, caption="Detected Signs")

    # Optional: Print the labels below clearly
    boxes = results[0].boxes
    if len(boxes) > 0:
        found_classes = [model.names[int(cls)] for cls in boxes.cls]
        unique_signs = set(found_classes)
        st.success(f"Detected: {', '.join(unique_signs)}")
    else:
        st.warning("No signs detected.")