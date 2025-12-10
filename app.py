import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="RoadGuard AI | Morocco",
    page_icon="🛑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ENHANCED CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Main container card */
    .main-card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 2rem 0 3rem 0;
        position: relative;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #D32F2F 0%, #F44336 50%, #FF5722 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #666;
        font-weight: 400;
        margin-bottom: 1rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 1rem;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Camera section */
    .camera-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        border: 3px dashed #667eea;
        transition: all 0.3s ease;
    }
    
    .camera-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .camera-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    
    /* Results section */
    .results-header {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        color: #333;
        margin: 3rem 0 2rem 0;
        position: relative;
        display: inline-block;
        width: 100%;
    }
    
    .results-header:after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #D32F2F, #F44336);
        border-radius: 2px;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        text-align: center;
    }
    
    div[data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Image container */
    .detected-image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .detected-image-container:hover {
        transform: scale(1.02);
    }
    
    /* Data table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Alert boxes */
    .stSuccess, .stWarning, .stInfo, .stError {
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: white;
        font-size: 0.95rem;
        margin-top: 3rem;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .footer-link {
        color: white;
        text-decoration: none;
        opacity: 0.8;
        transition: opacity 0.3s;
    }
    
    .footer-link:hover {
        opacity: 1;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #666;
        font-weight: 600;
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
        }
        50% {
            opacity: 0.5;
        }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Model not found! Please upload 'best.pt'. Error: {e}")
    st.stop()

# --- 4. ENHANCED SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")
    st.markdown("---")
    
    st.markdown("### 🎯 Detection Settings")
    confidence_threshold = st.slider(
        "AI Confidence Level", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.40,
        help="Minimum confidence score for sign detection"
    )
    
    st.markdown(f"**Current Threshold:** `{confidence_threshold:.0%}`")
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.info("**Model:** YOLOv8\n\n**Type:** Object Detection\n\n**Training:** Moroccan Traffic Signs")
    
    st.markdown("---")
    st.markdown("### 🇲🇦 About")
    st.markdown("Built with ❤️ for Moroccan road safety. This AI system detects and classifies traffic signs in real-time.")

# --- 5. MAIN UI WITH ENHANCED DESIGN ---
# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🛑 RoadGuard AI</div>
        <div class="hero-subtitle">Advanced Traffic Sign Detection for Moroccan Roads</div>
        <div class="hero-badge">🇲🇦 Powered by Artificial Intelligence</div>
    </div>
""", unsafe_allow_html=True)

# Main content container
col_main1, col_main2, col_main3 = st.columns([1, 6, 1])

with col_main2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Camera Section
    st.markdown("""
        <div class="camera-container">
            <div class="camera-title">
                <span>📸</span> Capture Traffic Sign
            </div>
            <p style="color: #666; margin-bottom: 1.5rem;">
                Point your camera at a traffic sign and take a photo for instant AI analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    img_buffer = st.camera_input("Take a photo", label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. PROCESSING & RESULTS ---
if img_buffer is not None:
    with st.spinner("🤖 AI is analyzing your image..."):
        # Load and process image
        image = Image.open(img_buffer)
        results = model.predict(image, conf=confidence_threshold)
        res_plotted = results[0].plot()
        boxes = results[0].boxes
        num_signs = len(boxes)
    
    # Results Section
    st.markdown('<div class="results-header">🔍 Detection Results</div>', unsafe_allow_html=True)
    
    # Create layout for results
    result_col1, result_col2, result_col3 = st.columns([1, 6, 1])
    
    with result_col2:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        
        # Stats row
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.metric(
                label="🎯 Signs Found", 
                value=num_signs,
                delta="Active" if num_signs > 0 else "None"
            )
        
        with stat_col2:
            avg_conf = sum([float(box.conf[0]) for box in boxes]) / len(boxes) if len(boxes) > 0 else 0
            st.metric(
                label="📊 Avg. Confidence",
                value=f"{avg_conf:.1%}",
                delta="High" if avg_conf > 0.7 else "Medium" if avg_conf > 0.4 else "Low"
            )
        
        with stat_col3:
            st.metric(
                label="⚡ Status",
                value="Ready",
                delta="Online"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Image and details
        img_col, details_col = st.columns([3, 2])
        
        with img_col:
            st.markdown('<div class="detected-image-container">', unsafe_allow_html=True)
            st.image(res_plotted, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with details_col:
            if num_signs > 0:
                st.success("✅ Detection Successful")
                
                st.markdown("### 🏷️ Identified Signs")
                
                # Extract detection data
                detected_data = []
                for i, box in enumerate(boxes, 1):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls_id]
                    detected_data.append({
                        "#": i,
                        "Sign Type": name,
                        "Confidence": f"{conf:.1%}"
                    })
                
                # Display as dataframe
                st.dataframe(
                    detected_data, 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Results",
                    data=str(detected_data),
                    file_name="detection_results.txt",
                    mime="text/plain"
                )
                
            else:
                st.warning("⚠️ No Signs Detected")
                st.markdown("""
                    **Tips for better detection:**
                    - Move closer to the sign
                    - Ensure good lighting
                    - Keep the sign centered
                    - Avoid motion blur
                """)
        
        # Technical details expander
        if num_signs > 0:
            with st.expander("🔬 Technical Details & Raw Data"):
                st.json({
                    "total_detections": num_signs,
                    "confidence_threshold": confidence_threshold,
                    "average_confidence": f"{avg_conf:.2%}",
                    "detections": detected_data
                })
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. FOOTER ---
st.markdown("""
    <div class="footer">
        <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem;">
            RoadGuard AI © 2024
        </div>
        <div>
            Making Moroccan roads safer with artificial intelligence
        </div>
        <div class="footer-links">
            <span class="footer-link">🤖 YOLOv8 Technology</span>
            <span class="footer-link">🇲🇦 Made in Morocco</span>
            <span class="footer-link">🔒 Privacy Focused</span>
            <span class="footer-link">⚡ Real-time Processing</span>
        </div>
    </div>
""", unsafe_allow_html=True)
