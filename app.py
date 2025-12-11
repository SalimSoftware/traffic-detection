import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time
from gtts import gTTS
import io
import base64
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Traffic Sign Detection | ML Project",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ENHANCED MODERN CSS (Your Original CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .hero-title { font-family: 'Space Grotesk', sans-serif; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1600px; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); background-attachment: fixed; }
    .glass-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 2rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); }
    .hero-section { text-align: center; padding: 2rem 0 1rem 0; margin-bottom: 2rem; }
    .hero-title { font-size: 4.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; letter-spacing: -3px; }
    .hero-subtitle { font-size: 1.5rem; color: rgba(255, 255, 255, 0.8); font-weight: 400; margin-bottom: 1.5rem; }
    .team-badge { background: rgba(102, 126, 234, 0.15); border: 1px solid rgba(102, 126, 234, 0.3); backdrop-filter: blur(10px); color: #a8b3ff; padding: 12px 24px; border-radius: 50px; font-size: 0.95rem; font-weight: 600; display: inline-block; margin: 0.5rem; }
    .status-indicator { display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 50px; font-size: 0.9rem; font-weight: 600; }
    .status-active { background: rgba(76, 175, 80, 0.2); color: #81c784; border: 1px solid rgba(76, 175, 80, 0.3); }
    .status-inactive { background: rgba(158, 158, 158, 0.2); color: #bdbdbd; border: 1px solid rgba(158, 158, 158, 0.3); }
    .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: #81c784; animation: pulse 2s ease-in-out infinite; }
    div[data-testid="stMetric"] { background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); border: 1px solid rgba(102, 126, 234, 0.3); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 16px; text-align: center; }
    div[data-testid="stMetric"] label { color: rgba(255, 255, 255, 0.7) !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: white !important; font-size: 2.8rem !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); border-right: 1px solid rgba(102, 126, 234, 0.2); }
    [data-testid="stSidebar"] * { color: rgba(255, 255, 255, 0.9) !important; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600; transition: all 0.3s ease; }
    .footer { text-align: center; padding: 3rem 0 1rem 0; color: rgba(255, 255, 255, 0.6); border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 4rem; }
    .team-info { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 2rem; margin: 2rem 0; }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.7; transform: scale(1.1); } }
    h1, h2, h3 { color: white !important; }
    p, span, div { color: rgba(255, 255, 255, 0.8); }
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
    except Exception as e:
        return None

def play_audio(audio_base64):
    audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Model file 'best.pt' not found. Error: {e}")
    st.stop()

# --- 5. SESSION STATE ---
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = 'single'
if 'vocal_alerts' not in st.session_state:
    st.session_state.vocal_alerts = True

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")
    st.markdown("---")
    
    st.markdown("### 🎯 Detection Mode")
    detection_mode = st.radio(
        "Select mode:",
        ["Single Shot", "Continuous Detection"],
        index=0 if st.session_state.detection_mode == 'single' else 1,
        label_visibility="collapsed"
    )
    st.session_state.detection_mode = 'continuous' if detection_mode == "Continuous Detection" else 'single'
    
    st.markdown("---")
    st.markdown("### 🔧 Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.40)
    
    st.markdown("---")
    st.markdown("### 🔊 Audio Alerts")
    st.session_state.vocal_alerts = st.toggle("Enable Vocal Alerts", value=True)
    
    st.markdown("---")
    st.markdown("### 👥 Project Team")
    st.markdown("**Developers:** Marwane, Salim, Saad\n**Supervisor:** Dr. Yousra Chtouki")

# --- 7. MAIN INTERFACE ---
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🚦 Traffic Sign Detection</div>
        <div class="hero-subtitle">AI-Powered Recognition System for Moroccan Road Safety</div>
        <div class="team-badge">Machine Learning Project Class</div>
        <div class="team-badge">🎓 Supervised by Dr. Yousra Chtouki</div>
    </div>
""", unsafe_allow_html=True)

# Status Indicator
col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
with col_s2:
    mode_text = "🔄 Continuous Live" if st.session_state.detection_mode == 'continuous' else "📸 Single Shot"
    st.markdown(f'<div style="text-align: center; margin-bottom: 2rem;"><span class="status-indicator status-active"><span class="pulse-dot"></span>{mode_text} Mode Active</span></div>', unsafe_allow_html=True)

# --- SINGLE SHOT MODE ---
if st.session_state.detection_mode == 'single':
    col_main1, col_main2, col_main3 = st.columns([1, 6, 1])
    with col_main2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📸 Capture Image")
        img_buffer = st.camera_input("Camera", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if img_buffer is not None:
            with st.spinner("🤖 Analyzing image..."):
                image = Image.open(img_buffer)
                results = model.predict(image, conf=confidence_threshold)
                res_plotted = results[0].plot()
                boxes = results[0].boxes
                
                # Audio Alert
                if len(boxes) > 0 and st.session_state.vocal_alerts:
                    sign_names = [model.names[int(box.cls[0])] for box in boxes]
                    alert_text = f"Warning! {len(boxes)} traffic sign detected: {sign_names[0]}"
                    audio = text_to_speech(alert_text)
                    if audio: play_audio(audio)
            
            # Display Results
            st.markdown("---")
            st.markdown("## 🔍 Detection Results")
            col1, col2 = st.columns([3, 2])
            with col1:
                st.image(res_plotted, use_container_width=True)
            with col2:
                if len(boxes) > 0:
                    st.success("✅ Signs Identified")
                    for box in boxes:
                        name = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        st.markdown(f"**{name}** ({conf:.1%})")
                else:
                    st.warning("⚠️ No signs detected")

# --- CONTINUOUS MODE (UPDATED WITH WEBRTC) ---
else:
    col_cont1, col_cont2, col_cont3 = st.columns([1, 8, 1])
    with col_cont2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔄 Real-Time Live Feed")
        st.info("The AI will draw boxes automatically on the video stream below.")
        
        # WEBRTC CALLBACK FUNCTION
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24") # Convert frame to numpy
            
            # Run YOLO on the frame
            results = model.predict(img, conf=confidence_threshold)
            
            # Draw boxes on the frame
            annotated_frame = results[0].plot()
            
            # Return the processed frame to the video stream
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        # START THE STREAM
        webrtc_streamer(
            key="continuous_detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# --- 8. FOOTER ---
st.markdown("""
    <div class="footer">
        <div class="team-info">
            <h3 style="margin-top: 0;">🎓 Academic Project Information</h3>
            <p><strong>Traffic Sign Detection System</strong> • Machine Learning Project Class</p>
            <p><strong>Project Team:</strong> Marwane, Salim, Saad • <strong>Supervised by:</strong> Dr. Yousra Chtouki</p>
            <p style="font-size: 0.9rem; opacity: 0.7;">🤖 Powered by YOLOv8 • 🇲🇦 Morocco</p>
        </div>
    </div>
""", unsafe_allow_html=True)
