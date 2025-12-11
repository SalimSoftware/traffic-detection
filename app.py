import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import base64
import cv2
import av
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Sign Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LUXURY CSS (FIXED) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Montserrat:wght@300;400;500;600&display=swap');
    
    * { font-family: 'Montserrat', sans-serif; }
    
    /* FIX: Make sure the sidebar button is visible */
    header[data-testid="stHeader"] {
        background: transparent;
        z-index: 999;
    }
    
    /* Color the hamburger menu white so it is visible on black */
    [data-testid="stSidebarNav"] {
        background-color: transparent;
    }
    
    .stApp { background: #000000; }
    
    /* Hero Styling */
    .hero-section {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 1px solid #333;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        color: white;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .hero-subtitle {
        color: #888;
        text-transform: uppercase;
        letter-spacing: 4px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: transparent;
        border: 1px solid #444;
        color: white;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 1rem 0;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        border-color: white;
        background: #111;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background: #111;
        border: 1px solid #333;
        text-align: center;
        padding: 1rem;
    }
    div[data-testid="stMetric"] label { color: #888; }
    div[data-testid="stMetric"] div { color: white !important; }
    
    /* Camera Input Styling */
    [data-testid="stCameraInput"] {
        border: 1px dashed #444;
        background: #050505;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #222;
    }
    [data-testid="stSidebar"] * { color: #ddd !important; }
    
    /* Hide default footer but keep header visible for menu */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        return base64.b64encode(audio_bytes).decode()
    except:
        return None

def play_audio(audio_base64):
    st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}"></audio>', unsafe_allow_html=True)

# --- 4. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 5. SESSION STATE ---
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = 'single'
if 'vocal_alerts' not in st.session_state:
    st.session_state.vocal_alerts = True
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.40

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Mode Switcher in Sidebar
    mode = st.radio("Detection Mode", ["Single Shot", "Continuous Live"], index=0 if st.session_state.detection_mode == 'single' else 1)
    st.session_state.detection_mode = 'single' if mode == "Single Shot" else 'continuous'
    
    st.divider()
    
    st.session_state.confidence = st.slider("Confidence Threshold", 0.0, 1.0, st.session_state.confidence)
    
    st.divider()
    
    st.session_state.vocal_alerts = st.toggle("Vocal Alerts", value=st.session_state.vocal_alerts)
    
    st.info("Developed by Marwane, Salim, Saad")

# --- 7. MAIN UI ---
st.markdown('<div class="hero-section"><div class="hero-title">Traffic Sign Detection</div><div class="hero-subtitle">Morocco AI Project</div></div>', unsafe_allow_html=True)

# Homepage Buttons (Quick Switch)
c1, c2 = st.columns(2)
if c1.button("📸 Single Photo Mode", use_container_width=True):
    st.session_state.detection_mode = 'single'
    st.rerun()
if c2.button("🔄 Continuous Video Mode", use_container_width=True):
    st.session_state.detection_mode = 'continuous'
    st.rerun()

st.divider()

# --- MODE 1: SINGLE SHOT ---
if st.session_state.detection_mode == 'single':
    st.subheader("📸 Single Shot Capture")
    img_buffer = st.camera_input("Take a photo", label_visibility="collapsed")
    
    if img_buffer:
        image = Image.open(img_buffer)
        results = model.predict(image, conf=st.session_state.confidence)
        res_plotted = results[0].plot()
        boxes = results[0].boxes
        
        # Audio Alert
        if len(boxes) > 0 and st.session_state.vocal_alerts:
            name = model.names[int(boxes[0].cls[0])]
            play_audio(text_to_speech(f"Detected {name}"))
        
        # Show Results
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(res_plotted, use_container_width=True)
        with col2:
            st.metric("Signs Found", len(boxes))
            if len(boxes) > 0:
                for box in boxes:
                    st.success(f"{model.names[int(box.cls[0])]}")

# --- MODE 2: CONTINUOUS LIVE VIDEO (WEBRTC) ---
else:
    st.subheader("🔄 Real-Time Live Stream")
    st.warning("Ensure you have granted camera permissions.")

    # Webrtc Callback
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inference
        results = model.predict(img, conf=0.40) # Use fixed conf for speed or pass global
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Streamer
    webrtc_streamer(
        key="traffic-sign-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
