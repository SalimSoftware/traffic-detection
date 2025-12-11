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
    initial_sidebar_state="expanded"
)

# --- 2. LUXURY BLACK & WHITE CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;800&family=Montserrat:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Montserrat', sans-serif;
        -webkit-tap-highlight-color: transparent;
    }
    
    h1, h2, h3, .hero-title {
        font-family: 'Playfair Display', serif;
    }
    
    /* Mobile-first layout */
    .block-container {
        padding: 1rem 1rem 2rem 1rem !important;
        max-width: 100% !important;
    }
    
    /* Luxury black background */
    .stApp {
        background: #000000;
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #2a2a2a;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .hero-subtitle {
        font-size: 0.9rem;
        color: #999999;
        font-weight: 300;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .hero-divider {
        width: 60px;
        height: 1px;
        background: #ffffff;
        margin: 1rem auto;
    }
    
    .team-badge {
        background: transparent;
        border: 1px solid #333333;
        color: #cccccc;
        padding: 6px 14px;
        border-radius: 2px;
        font-size: 0.7rem;
        font-weight: 400;
        display: inline-block;
        margin: 0.3rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Mode toggle buttons */
    .stButton > button {
        background: transparent;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 0;
        padding: 1.2rem;
        font-weight: 500;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        font-family: 'Montserrat', sans-serif;
    }
    
    .stButton > button:hover {
        background: #ffffff;
        color: #000000;
        border-color: #ffffff;
    }
    
    .stButton > button[kind="primary"] {
        background: #ffffff;
        color: #000000;
        border: 1px solid #ffffff;
    }
    
    /* Status bar */
    .status-bar {
        background: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 0;
        padding: 1rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .status-active {
        color: #ffffff;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .status-dot {
        width: 6px;
        height: 6px;
        background: #ffffff;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Camera card */
    .camera-card {
        background: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 0;
        padding: 2rem 1rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    
    .camera-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .camera-subtitle {
        font-size: 0.75rem;
        color: #666666;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }
    
    /* Camera input */
    [data-testid="stCameraInput"] {
        border-radius: 0;
        background: #000000;
        border: 1px solid #222222;
    }
    
    [data-testid="stCameraInput"] button {
        width: 100%;
        padding: 1.2rem;
        font-size: 0.9rem;
        font-weight: 500;
        border-radius: 0;
        background: #ffffff;
        border: 1px solid #ffffff;
        color: #000000;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-family: 'Montserrat', sans-serif;
    }
    
    [data-testid="stCameraInput"] button:hover {
        background: #000000;
        color: #ffffff;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 0;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stMetric"] label {
        color: #666666 !important;
        font-size: 0.75rem !important;
        font-weight: 400 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 300 !important;
        font-family: 'Playfair Display', serif !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #999999 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    /* Image display */
    .image-display {
        border: 1px solid #222222;
        border-radius: 0;
        overflow: hidden;
        margin: 1.5rem 0;
        background: #0a0a0a;
    }
    
    /* Results card */
    .results-card {
        background: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 0;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .results-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
        text-align: center;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-family: 'Playfair Display', serif;
    }
    
    /* Detection items */
    .detection-item {
        background: #000000;
        border: 1px solid #1a1a1a;
        border-radius: 0;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .detection-number {
        font-size: 0.7rem;
        color: #666666;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .detection-name {
        font-weight: 500;
        color: #ffffff;
        font-size: 1rem;
        letter-spacing: 1px;
    }
    
    .detection-conf {
        background: transparent;
        border: 1px solid #333333;
        padding: 6px 14px;
        border-radius: 0;
        font-size: 0.75rem;
        color: #cccccc;
        font-weight: 400;
        letter-spacing: 1px;
    }
    
    /* Alerts */
    .stSuccess {
        background: #0a0a0a;
        border: 1px solid #ffffff;
        border-radius: 0;
        color: #ffffff !important;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .stWarning {
        background: #0a0a0a;
        border: 1px solid #666666;
        border-radius: 0;
        color: #cccccc !important;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .stInfo {
        background: #0a0a0a;
        border: 1px solid #333333;
        border-radius: 0;
        color: #999999 !important;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #000000;
        border-right: 1px solid #222222;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h2 {
        font-family: 'Playfair Display', serif;
        letter-spacing: 2px;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    [data-testid="stSidebar"] h3 {
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
        color: #cccccc !important;
        margin-top: 1.5rem;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #ffffff;
    }
    
    .stSlider > div > div > div > div {
        background: #ffffff;
    }
    
    /* Toggle */
    .stCheckbox {
        background: #0a0a0a;
        border: 1px solid #222222;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #0a0a0a;
        border: 1px solid #222222;
        padding: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 0;
        color: #ffffff !important;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 0.8rem;
    }
    
    /* Footer */
    .footer {
        background: #0a0a0a;
        border: 1px solid #222222;
        border-top: 2px solid #ffffff;
        border-radius: 0;
        padding: 2rem 1rem;
        margin: 3rem 0 1rem 0;
        text-align: center;
    }
    
    .footer-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-family: 'Playfair Display', serif;
    }
    
    .footer-text {
        font-size: 0.75rem;
        color: #666666;
        line-height: 1.8;
        letter-spacing: 1px;
    }
    
    .footer-divider {
        width: 40px;
        height: 1px;
        background: #333333;
        margin: 1rem auto;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.4;
        }
    }
    
    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Text colors */
    p, span, div, label {
        color: #cccccc;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #222222;
        margin: 2rem 0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def text_to_speech(text):
    """Convert text to speech in French"""
    try:
        tts = gTTS(text=text, lang='fr', slow=False)
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
    st.error(f"⚠️ Modèle introuvable: {e}")
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

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")
    
    st.markdown("### 🎯 Mode de Détection")
    mode_choice = st.radio(
        "Choisir le mode:",
        ["Photo Unique", "Détection Continue"],
        index=0 if st.session_state.detection_mode == 'single' else 1
    )
    st.session_state.detection_mode = 'single' if mode_choice == "Photo Unique" else 'continuous'
    
    st.markdown("---")
    
    st.markdown("### 🔧 Seuil de Confiance")
    st.session_state.confidence = st.slider(
        "Niveau de confiance", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.confidence,
        help="Seuil minimum pour la détection"
    )
    
    st.markdown(f"**Actuel:** `{st.session_state.confidence:.0%}`")
    
    st.markdown("---")
    
    st.markdown("### 🔊 Alertes Vocales")
    st.markdown("*(Uniquement en mode Photo Unique)*")
    st.session_state.vocal_alerts = st.toggle(
        "Activer les alertes vocales",
        value=st.session_state.vocal_alerts,
        disabled=(st.session_state.detection_mode == 'continuous')
    )
    
    if st.session_state.detection_mode == 'single' and st.session_state.vocal_alerts:
        st.success("🔊 Alertes: Activées")
    else:
        st.info("🔇 Alertes: Désactivées")
    
    st.markdown("---")
    
    st.markdown("### 📊 Modèle")
    st.info("""
    **Architecture:** YOLOv8  
    **Type:** Détection d'objets  
    **Dataset:** Panneaux marocains
    """)
    
    st.markdown("---")
    
    st.markdown("### 👥 Équipe")
    st.markdown("""
    **Développeurs:**  
    Marwane, Salim, Saad
    
    **Superviseur:**  
    Dr. Yousra Chtouki
    
    **Cours:**  
    Projet Machine Learning
    """)

# --- 7. MAIN INTERFACE ---

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Détection de Panneaux</div>
        <div class="hero-divider"></div>
        <div class="hero-subtitle">Intelligence Artificielle</div>
        <div class="team-badge">Projet ML</div>
        <div class="team-badge">Dr. Yousra Chtouki</div>
    </div>
""", unsafe_allow_html=True)

# Info tip
st.info("💡 Utilisez le menu ☰ (en haut à gauche) pour accéder aux paramètres")

# Mode Selection
col1, col2 = st.columns(2)

with col1:
    if st.button("📸 Photo Unique", use_container_width=True, 
                 type="primary" if st.session_state.detection_mode == 'single' else "secondary",
                 key="btn_single"):
        st.session_state.detection_mode = 'single'
        st.rerun()

with col2:
    if st.button("🔄 Continue", use_container_width=True,
                 type="primary" if st.session_state.detection_mode == 'continuous' else "secondary",
                 key="btn_continuous"):
        st.session_state.detection_mode = 'continuous'
        st.rerun()

# Status Bar
mode_text = "Surveillance Continue" if st.session_state.detection_mode == 'continuous' else "Mode Photo Unique"
alert_status = "🔊 Activées" if (st.session_state.vocal_alerts and st.session_state.detection_mode == 'single') else "🔇 Désactivées"

st.markdown(f"""
    <div class="status-bar">
        <span class="status-active">
            <span class="status-dot"></span>
            {mode_text}
        </span>
        <span style="color: #666; font-size: 0.75rem;">{alert_status}</span>
    </div>
""", unsafe_allow_html=True)

# --- CAMERA INPUT ---
st.markdown("""
    <div class="camera-card">
        <div class="camera-title">Capture d'Image</div>
        <div class="camera-subtitle">Pointez vers un panneau et capturez</div>
    </div>
""", unsafe_allow_html=True)

img_buffer = st.camera_input("Caméra", label_visibility="collapsed")

# --- PROCESSING ---
if img_buffer is not None:
    with st.spinner("Analyse en cours..."):
        image = Image.open(img_buffer)
        results = model.predict(image, conf=st.session_state.confidence)
        res_plotted = results[0].plot()
        boxes = results[0].boxes
        num_signs = len(boxes)
        
        # Vocal alert ONLY for single shot mode
        if num_signs > 0 and st.session_state.vocal_alerts and st.session_state.detection_mode == 'single':
            sign_names = [model.names[int(box.cls[0])] for box in boxes]
            alert_text = f"Attention! {num_signs} panneau détecté: {sign_names[0]}"
            audio = text_to_speech(alert_text)
            if audio:
                play_audio(audio)
    
    # --- RESULTS ---
    st.markdown('<div class="results-card">', unsafe_allow_html=True)
    st.markdown('<div class="results-title">Résultats</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics
    st.metric("🎯 Panneaux Détectés", num_signs, 
             delta="Actif" if num_signs > 0 else "Aucun")
    
    if num_signs > 0:
        avg_conf = sum([float(box.conf[0]) for box in boxes]) / len(boxes)
        st.metric("📊 Confiance Moyenne", f"{avg_conf:.1%}")
    
    # Image Display
    st.markdown('<div class="image-display">', unsafe_allow_html=True)
    st.image(res_plotted, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detection Details
    if num_signs > 0:
        st.success("✓ Panneaux identifiés avec succès")
        
        st.markdown("### Panneaux Détectés")
        
        for i, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            
            st.markdown(f"""
                <div class="detection-item">
                    <div>
                        <div class="detection-number">Panneau #{i}</div>
                        <div class="detection-name">{name}</div>
                    </div>
                    <div class="detection-conf">{conf:.0%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Technical Details
        with st.expander("Détails Techniques"):
            detected_data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                detected_data.append({
                    "panneau": name,
                    "confiance": f"{conf:.2%}"
                })
            
            st.json({
                "total": num_signs,
                "seuil": st.session_state.confidence,
                "confiance_moyenne": f"{avg_conf:.2%}",
                "alertes_vocales": st.session_state.vocal_alerts and st.session_state.detection_mode == 'single',
                "détections": detected_data
            })
    else:
        st.warning("Aucun panneau détecté")
        st.info("""
        **Conseils:**
        - Rapprochez-vous du panneau
        - Assurez un bon éclairage
        - Centrez le panneau
        - Stabilisez la caméra
        """)

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        <div class="footer-title">Projet de Détection de Panneaux</div>
        <div class="footer-divider"></div>
        <div class="footer-text">
            <strong>Équipe:</strong> Marwane, Salim, Saad<br>
            <strong>Superviseur:</strong> Dr. Yousra Chtouki<br>
            Projet Machine Learning<br><br>
            YOLOv8 • Maroc • 2024
        </div>
    </div>
""", unsafe_allow_html=True)
