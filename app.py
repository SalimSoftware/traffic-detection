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
    page_title="Traffic Sign Detection | ML Project",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ENHANCED MODERN CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .hero-title {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }
    
    /* Dark modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 4.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -3px;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 400;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Team badge */
    .team-badge {
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        color: #a8b3ff;
        padding: 12px 24px;
        border-radius: 50px;
        font-size: 0.95rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Mode toggle buttons */
    .mode-container {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 2rem 0;
    }
    
    /* Detection modes */
    .detection-mode-card {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .detection-mode-card:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #667eea;
        transform: translateY(-5px);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .status-active {
        background: rgba(76, 175, 80, 0.2);
        color: #81c784;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .status-inactive {
        background: rgba(158, 158, 158, 0.2);
        color: #bdbdbd;
        border: 1px solid rgba(158, 158, 158, 0.3);
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #81c784;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
    }
    
    div[data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Alert boxes */
    .stSuccess {
        background: rgba(76, 175, 80, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        color: #81c784 !important;
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.15);
        border: 1px solid rgba(255, 152, 0, 0.3);
        border-radius: 12px;
        color: #ffb74d !important;
    }
    
    .stInfo {
        background: rgba(33, 150, 243, 0.15);
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 12px;
        color: #64b5f6 !important;
    }
    
    /* Image container */
    .image-display {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Data table */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.5);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Camera input */
    [data-testid="stCameraInput"] {
        border-radius: 16px;
        border: 2px dashed rgba(102, 126, 234, 0.4);
        background: rgba(255, 255, 255, 0.03);
        padding: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0 1rem 0;
        color: rgba(255, 255, 255, 0.6);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 4rem;
    }
    
    .team-info {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.7;
            transform: scale(1.1);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        }
        50% {
            box-shadow: 0 0 30px rgba(102, 126, 234, 0.6);
        }
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Text colors */
    h1, h2, h3 {
        color: white !important;
    }
    
    p, span, div {
        color: rgba(255, 255, 255, 0.8);
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
        st.error(f"Audio generation error: {e}")
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
    st.error(f"⚠️ Model file 'best.pt' not found. Error: {e}")
    st.stop()

# --- 5. SESSION STATE INITIALIZATION ---
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = 'single'
if 'continuous_active' not in st.session_state:
    st.session_state.continuous_active = False
if 'last_detection' not in st.session_state:
    st.session_state.last_detection = None
if 'vocal_alerts' not in st.session_state:
    st.session_state.vocal_alerts = True

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")
    st.markdown("---")
    
    # Detection Mode Selection
    st.markdown("### 🎯 Detection Mode")
    detection_mode = st.radio(
        "Select mode:",
        ["Single Shot", "Continuous Detection"],
        index=0 if st.session_state.detection_mode == 'single' else 1,
        label_visibility="collapsed"
    )
    st.session_state.detection_mode = 'continuous' if detection_mode == "Continuous Detection" else 'single'
    
    st.markdown("---")
    
    # Settings
    st.markdown("### 🔧 Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.40,
        help="Minimum confidence for detection"
    )
    
    st.markdown(f"**Current:** `{confidence_threshold:.0%}`")
    
    st.markdown("---")
    
    # Audio Settings
    st.markdown("### 🔊 Audio Alerts")
    st.session_state.vocal_alerts = st.toggle(
        "Enable Vocal Alerts",
        value=True,
        help="Play audio when signs are detected"
    )
    
    if st.session_state.vocal_alerts:
        st.success("🔊 Alerts: ON")
    else:
        st.info("🔇 Alerts: OFF")
    
    st.markdown("---")
    
    # Model Info
    st.markdown("### 📊 Model Info")
    st.info("""
    **Architecture:** YOLOv8  
    **Task:** Object Detection  
    **Dataset:** Moroccan Traffic Signs  
    **Classes:** Multiple sign types
    """)
    
    st.markdown("---")
    
    # Team Information
    st.markdown("### 👥 Project Team")
    st.markdown("""
    **Developers:**
    - 👨‍💻 Marwane
    - 👨‍💻 Salim  
    - 👨‍💻 Saad
    
    **Supervisor:**  
    🎓 Dr. Yousra Chtouki
    
    **Course:**  
    Machine Learning Project
    """)

# --- 7. MAIN INTERFACE ---

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🚦 Traffic Sign Detection</div>
        <div class="hero-subtitle">AI-Powered Recognition System for Moroccan Road Safety</div>
        <div class="team-badge">Machine Learning Project Class</div>
        <div class="team-badge">🎓 Supervised by Dr. Yousra Chtouki</div>
    </div>
""", unsafe_allow_html=True)

# Mode Status Display
col_status1, col_status2, col_status3 = st.columns([1, 2, 1])
with col_status2:
    mode_display = "🔄 Continuous" if st.session_state.detection_mode == 'continuous' else "📸 Single Shot"
    status_class = "status-active" if st.session_state.detection_mode == 'continuous' else "status-inactive"
    
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="status-indicator {status_class}">
                <span class="pulse-dot"></span>
                {mode_display} Mode Active
            </span>
        </div>
    """, unsafe_allow_html=True)

# Main Content Area
if st.session_state.detection_mode == 'single':
    # --- SINGLE SHOT MODE ---
    col_main1, col_main2, col_main3 = st.columns([1, 6, 1])
    
    with col_main2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📸 Capture Image")
        st.markdown("Take a photo of a traffic sign for instant AI analysis")
        
        img_buffer = st.camera_input("Camera", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if img_buffer is not None:
            with st.spinner("🤖 Analyzing image..."):
                # Process image
                image = Image.open(img_buffer)
                results = model.predict(image, conf=confidence_threshold)
                res_plotted = results[0].plot()
                boxes = results[0].boxes
                num_signs = len(boxes)
                
                # Play vocal alert if enabled
                if num_signs > 0 and st.session_state.vocal_alerts:
                    sign_names = [model.names[int(box.cls[0])] for box in boxes]
                    alert_text = f"Warning! {num_signs} traffic sign detected: {sign_names[0]}"
                    audio = text_to_speech(alert_text)
                    if audio:
                        play_audio(audio)
            
            # Display results
            st.markdown("---")
            st.markdown("## 🔍 Detection Results")
            
            # Metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("🎯 Signs Detected", num_signs, 
                         delta="Active" if num_signs > 0 else "None")
            
            with metric_col2:
                avg_conf = sum([float(box.conf[0]) for box in boxes]) / len(boxes) if len(boxes) > 0 else 0
                st.metric("📊 Avg Confidence", f"{avg_conf:.1%}",
                         delta="High" if avg_conf > 0.7 else "Medium")
            
            with metric_col3:
                st.metric("⚡ Processing", "Complete", delta="Success")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Image and details
            result_col1, result_col2 = st.columns([3, 2])
            
            with result_col1:
                st.markdown('<div class="image-display">', unsafe_allow_html=True)
                st.image(res_plotted, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col2:
                if num_signs > 0:
                    st.success("✅ Signs Identified")
                    
                    # Detection data
                    detected_data = []
                    for i, box in enumerate(boxes, 1):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = model.names[cls_id]
                        detected_data.append({
                            "#": i,
                            "Sign": name,
                            "Confidence": f"{conf:.1%}"
                        })
                    
                    st.dataframe(detected_data, use_container_width=True, hide_index=True)
                    
                    # Technical details
                    with st.expander("🔬 Technical Details"):
                        st.json({
                            "total_detections": num_signs,
                            "threshold": confidence_threshold,
                            "avg_confidence": f"{avg_conf:.2%}",
                            "vocal_alert": st.session_state.vocal_alerts,
                            "detections": detected_data
                        })
                else:
                    st.warning("⚠️ No signs detected")
                    st.markdown("""
                    **Tips:**
                    - Move closer to sign
                    - Ensure good lighting
                    - Center the sign
                    - Reduce motion blur
                    """)

else:
    # --- CONTINUOUS DETECTION MODE ---
    col_cont1, col_cont2, col_cont3 = st.columns([1, 6, 1])
    
    with col_cont2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔄 Continuous Monitoring")
        st.markdown("Real-time detection with automatic vocal alerts")
        
        # Continuous detection camera
        img_buffer = st.camera_input("Live Feed", label_visibility="collapsed", key="continuous_cam")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if img_buffer is not None:
            # Process continuously
            image = Image.open(img_buffer)
            results = model.predict(image, conf=confidence_threshold)
            res_plotted = results[0].plot()
            boxes = results[0].boxes
            num_signs = len(boxes)
            
            # Vocal alert for new detections
            if num_signs > 0 and st.session_state.vocal_alerts:
                current_detection = [model.names[int(box.cls[0])] for box in boxes]
                
                # Only alert if detection changed
                if st.session_state.last_detection != current_detection:
                    alert_text = f"Alert! {num_signs} traffic sign detected: {', '.join(current_detection[:2])}"
                    audio = text_to_speech(alert_text)
                    if audio:
                        play_audio(audio)
                    st.session_state.last_detection = current_detection
            
            # Live results display
            st.markdown("## 📡 Live Detection Feed")
            
            live_col1, live_col2 = st.columns([2, 1])
            
            with live_col1:
                st.markdown('<div class="image-display">', unsafe_allow_html=True)
                st.image(res_plotted, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with live_col2:
                st.metric("🎯 Active Detections", num_signs)
                
                if num_signs > 0:
                    st.success("🔊 Alert Triggered")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        name = model.names[cls_id]
                        conf = float(box.conf[0])
                        st.markdown(f"**{name}** - `{conf:.0%}`")
                else:
                    st.info("👀 Monitoring...")

# --- 8. FOOTER ---
st.markdown("""
    <div class="footer">
        <div class="team-info">
            <h3 style="margin-top: 0;">🎓 Academic Project Information</h3>
            <p style="font-size: 1.1rem; margin: 1rem 0;">
                <strong>Traffic Sign Detection System</strong><br>
                Machine Learning Project Class
            </p>
            <p style="margin: 1rem 0;">
                <strong>Project Team:</strong> Marwane, Salim, Saad<br>
                <strong>Supervised by:</strong> Dr. Yousra Chtouki
            </p>
            <p style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.7;">
                🤖 Powered by YOLOv8 • 🇲🇦 Morocco • 🔬 Deep Learning
            </p>
        </div>
        <p style="margin-top: 2rem; opacity: 0.5;">
            © 2024 Traffic Sign Detection Project. All rights reserved.
        </p>
    </div>
""", unsafe_allow_html=True)
