import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
from gtts import gTTS
import io
import base64

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Sign Detection",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. MOBILE-OPTIMIZED CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        -webkit-tap-highlight-color: transparent;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Mobile-first layout */
    .block-container {
        padding: 1rem 1rem 2rem 1rem !important;
        max-width: 100% !important;
    }
    
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Hero section - compact for mobile */
    .hero-section {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 1rem;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
        margin-bottom: 0.8rem;
    }
    
    .team-badge {
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        color: #a8b3ff;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    /* Mode toggle - mobile optimized */
    .mode-toggle-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 0.5rem;
        display: flex;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .mode-btn {
        flex: 1;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .mode-btn-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid rgba(102, 126, 234, 0.5);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .mode-icon {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }
    
    .mode-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Status indicator */
    .status-bar {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 0.8rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .status-active {
        color: #81c784;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #81c784;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Camera card - full width mobile */
    .camera-card {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .camera-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .camera-subtitle {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        margin-bottom: 1rem;
    }
    
    /* Camera input styling */
    [data-testid="stCameraInput"] {
        border-radius: 12px;
        background: rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stCameraInput"] button {
        width: 100%;
        padding: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
    }
    
    /* Metric cards - stacked for mobile */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Image display - full width */
    .image-display {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        margin: 1rem 0;
    }
    
    /* Results card */
    .results-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .results-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Detection list */
    .detection-item {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .detection-name {
        font-weight: 600;
        color: white;
        font-size: 1rem;
    }
    
    .detection-conf {
        background: rgba(102, 126, 234, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #a8b3ff;
        font-weight: 600;
    }
    
    /* Alerts */
    .stSuccess, .stWarning, .stInfo {
        border-radius: 12px;
        padding: 1rem;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .stSuccess {
        background: rgba(76, 175, 80, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.3);
        color: #81c784 !important;
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.15);
        border: 1px solid rgba(255, 152, 0, 0.3);
        color: #ffb74d !important;
    }
    
    .stInfo {
        background: rgba(33, 150, 243, 0.15);
        border: 1px solid rgba(33, 150, 243, 0.3);
        color: #64b5f6 !important;
    }
    
    /* Settings button */
    .settings-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 1000;
        border: none;
        color: white;
        font-size: 1.5rem;
    }
    
    /* Footer - compact */
    .footer {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    .footer-title {
        font-size: 1rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .footer-text {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.6);
        line-height: 1.5;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.6;
            transform: scale(1.2);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Hide Streamlit branding on mobile */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Toggle styling */
    .stCheckbox > label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.15);
        border-radius: 12px;
        color: white !important;
        font-weight: 600;
    }
    
    /* Text color fixes */
    p, span, div, label {
        color: rgba(255, 255, 255, 0.9);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def text_to_speech(text):
    """Convert text to speech and return audio bytes"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        return audio_base64
    except Exception as e:
        return None

def play_audio(audio_base64):
    """Play audio using HTML audio element"""
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Model file not found: {e}")
    st.stop()

# --- 5. SESSION STATE ---
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = 'single'
if 'last_detection' not in st.session_state:
    st.session_state.last_detection = None
if 'vocal_alerts' not in st.session_state:
    st.session_state.vocal_alerts = True
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.40

# --- 6. SIDEBAR (Mobile Settings) ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    
    st.markdown("### 🔧 Detection")
    st.session_state.confidence = st.slider(
        "Confidence Level", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.confidence,
        help="Detection threshold"
    )
    
    st.markdown(f"**Current:** `{st.session_state.confidence:.0%}`")
    
    st.markdown("---")
    
    st.markdown("### 🔊 Audio")
    st.session_state.vocal_alerts = st.toggle(
        "Vocal Alerts",
        value=st.session_state.vocal_alerts
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Model")
    st.info("**YOLOv8**\nMoroccan Traffic Signs")
    
    st.markdown("---")
    
    st.markdown("### 👥 Team")
    st.markdown("""
    **Developers:**  
    Marwane, Salim, Saad
    
    **Supervisor:**  
    Dr. Yousra Chtouki
    
    **Course:**  
    ML Project Class
    """)

# --- 7. MAIN MOBILE INTERFACE ---

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🚦 Traffic Sign Detection</div>
        <div class="hero-subtitle">AI-Powered Mobile Recognition</div>
        <div class="team-badge">ML Project</div>
        <div class="team-badge">Dr. Yousra Chtouki</div>
    </div>
""", unsafe_allow_html=True)

# Mode Selection - Mobile Toggle
col1, col2 = st.columns(2)

with col1:
    if st.button("📸 Single Shot", use_container_width=True, 
                 type="primary" if st.session_state.detection_mode == 'single' else "secondary"):
        st.session_state.detection_mode = 'single'
        st.rerun()

with col2:
    if st.button("🔄 Continuous", use_container_width=True,
                 type="primary" if st.session_state.detection_mode == 'continuous' else "secondary"):
        st.session_state.detection_mode = 'continuous'
        st.rerun()

# Status Bar
mode_text = "🔄 Continuous Monitoring" if st.session_state.detection_mode == 'continuous' else "📸 Single Shot Mode"
alert_status = "🔊 ON" if st.session_state.vocal_alerts else "🔇 OFF"

st.markdown(f"""
    <div class="status-bar">
        <span class="status-active">
            <span class="pulse-dot"></span>
            {mode_text}
        </span>
        <span style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">{alert_status}</span>
    </div>
""", unsafe_allow_html=True)

# --- CAMERA INPUT ---
st.markdown("""
    <div class="camera-card">
        <div class="camera-title">📸 Capture Image</div>
        <div class="camera-subtitle">Point at a traffic sign and tap to capture</div>
    </div>
""", unsafe_allow_html=True)

img_buffer = st.camera_input("Camera", label_visibility="collapsed", key="mobile_cam")

# --- PROCESSING ---
if img_buffer is not None:
    with st.spinner("🤖 Analyzing..."):
        image = Image.open(img_buffer)
        results = model.predict(image, conf=st.session_state.confidence)
        res_plotted = results[0].plot()
        boxes = results[0].boxes
        num_signs = len(boxes)
        
        # Vocal alert
        if num_signs > 0 and st.session_state.vocal_alerts:
            current_detection = [model.names[int(box.cls[0])] for box in boxes]
            
            if st.session_state.detection_mode == 'continuous':
                if st.session_state.last_detection != current_detection:
                    alert_text = f"Alert! {num_signs} sign detected: {current_detection[0]}"
                    audio = text_to_speech(alert_text)
                    if audio:
                        play_audio(audio)
                    st.session_state.last_detection = current_detection
            else:
                alert_text = f"Warning! {num_signs} sign detected: {current_detection[0]}"
                audio = text_to_speech(alert_text)
                if audio:
                    play_audio(audio)
    
    # --- RESULTS ---
    st.markdown('<div class="results-card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="results-title">🔍 Detection Results</div>', unsafe_allow_html=True)
    
    # Metrics - Stacked for mobile
    st.metric("🎯 Signs Detected", num_signs, 
             delta="Active" if num_signs > 0 else "None")
    
    if num_signs > 0:
        avg_conf = sum([float(box.conf[0]) for box in boxes]) / len(boxes)
        st.metric("📊 Average Confidence", f"{avg_conf:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Image Display - Full Width
    st.markdown('<div class="image-display fade-in">', unsafe_allow_html=True)
    st.image(res_plotted, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detection Details
    if num_signs > 0:
        st.success("✅ Signs Identified Successfully")
        
        st.markdown("### 🏷️ Detected Signs")
        
        for i, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            
            st.markdown(f"""
                <div class="detection-item">
                    <div>
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-bottom: 4px;">#{i}</div>
                        <div class="detection-name">{name}</div>
                    </div>
                    <div class="detection-conf">{conf:.0%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Technical Details
        with st.expander("🔬 Technical Info"):
            detected_data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                detected_data.append({
                    "sign": name,
                    "confidence": f"{conf:.2%}"
                })
            
            st.json({
                "total": num_signs,
                "threshold": st.session_state.confidence,
                "avg_conf": f"{avg_conf:.2%}",
                "detections": detected_data
            })
    else:
        st.warning("⚠️ No signs detected")
        st.info("""
        **Tips:**
        - Move closer to the sign
        - Ensure good lighting
        - Center the sign in frame
        - Hold camera steady
        """)

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        <div class="footer-title">Traffic Sign Detection Project</div>
        <div class="footer-text">
            <strong>Team:</strong> Marwane, Salim, Saad<br>
            <strong>Supervisor:</strong> Dr. Yousra Chtouki<br>
            Machine Learning Project Class<br><br>
            🤖 YOLOv8 • 🇲🇦 Morocco • 2024
        </div>
    </div>
""", unsafe_allow_html=True)
