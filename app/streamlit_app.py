"""
app/streamlit_app.py
--------------------
Interactive Streamlit frontend for Kidney Disease Classification.

Calls the FastAPI backend over HTTP to get real-time predictions for uploaded
kidney CT scans.

Target Classes:
    - Normal
    - Cyst
    - Tumor
    - Stone
"""

import os
import time
from io import BytesIO
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CT Kidney Disease Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern aesthetics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.25rem;
        border-radius: 10px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        color: #991B1B;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    .badge-normal {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-disease {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Configuration & Backend Connection
# ---------------------------------------------------------------------------
DEFAULT_API_URL = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/kidney.png", width=70)
    st.title("Settings & Status")
    
    api_url = st.text_input(
        "FastAPI Backend URL",
        value=DEFAULT_API_URL,
        help="URL of the running FastAPI inference backend service.",
    ).rstrip("/")
    
    st.markdown("---")
    st.subheader("Backend Health")
    
    health_status: Optional[Dict] = None
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        if resp.status_code == 200:
            health_status = resp.json()
            st.success("🟢 Backend Connected & Ready")
            st.caption(f"Model Loaded: **{health_status.get('model_loaded', False)}**")
        else:
            st.warning(f"🟡 Backend returned HTTP {resp.status_code}")
    except requests.exceptions.RequestException:
        st.error("🔴 Cannot reach Backend API")
        st.caption(
            "Ensure the FastAPI service is running locally or that your Render service is awake."
        )

    st.markdown("---")
    st.subheader("Target Classes")
    st.markdown(
        """
        - 🟢 **Normal**: Healthy kidney tissue
        - 🟡 **Cyst**: Fluid-filled sac
        - 🔴 **Tumor**: Abnormal mass growth
        - 🟠 **Stone**: Solid mineral deposit
        """
    )


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🩺 CT Kidney Disease Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Deep Learning-assisted Kidney CT image classification using EfficientNetB0</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Upload Kidney CT Scan")
    uploaded_file = st.file_uploader(
        "Choose a CT scan image (JPEG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="Upload an axial CT scan slice of a kidney.",
    )

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            pil_img = Image.open(BytesIO(image_bytes))
            st.image(
                pil_img,
                caption=f"Uploaded: {uploaded_file.name} ({pil_img.size[0]}x{pil_img.size[1]} px)",
                use_column_width=True,
            )
        except Exception as e:
            st.error(f"Error loading image preview: {e}")

with col_right:
    st.subheader("2. Diagnostic Analysis")

    if uploaded_file is None:
        st.info("👈 Upload a kidney CT scan on the left to run classification.")
    else:
        analyze_btn = st.button("🔍 Run Classification", type="primary", use_container_width=True)

        if analyze_btn or "last_prediction" in st.session_state:
            with st.spinner("Analyzing CT scan with EfficientNetB0 backend..."):
                start_time = time.time()
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "image/jpeg",
                        )
                    }
                    response = requests.post(
                        f"{api_url}/predict",
                        files=files,
                        timeout=30,  # handles potential Render free-tier cold starts
                    )
                    latency = round(time.time() - start_time, 2)

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["last_prediction"] = data

                        prediction = data["prediction"]
                        confidence = data["confidence"] * 100
                        probabilities = data["probabilities"]

                        # Prediction Display
                        badge_style = "badge-normal" if prediction == "Normal" else "badge-disease"
                        
                        st.markdown(
                            f"""
                            <div class="prediction-card">
                                <h4 style="margin:0; color:#475569;">Predicted Class</h4>
                                <h1 style="margin:0.2rem 0; font-size:2.2rem;">
                                    <span class="{badge_style}">{prediction}</span>
                                </h1>
                                <p style="margin:0; font-size:1.1rem; color:#334155;">
                                    Confidence: <strong>{confidence:.2f}%</strong> 
                                    <span style="color:#94A3B8; font-size:0.85rem;">(Inference time: {latency}s)</span>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Probability Distribution Chart
                        st.markdown("#### Class Probability Distribution")
                        df_probs = pd.DataFrame(
                            list(probabilities.items()),
                            columns=["Class", "Probability"],
                        ).sort_values(by="Probability", ascending=False)
                        
                        # Show interactive bar chart
                        st.bar_chart(
                            data=df_probs.set_index("Class"),
                            y="Probability",
                            color="#3B82F6",
                            height=220,
                        )

                        # Detailed Probability Table
                        with st.expander("📊 View Detailed Probability Breakdown"):
                            df_display = df_probs.copy()
                            df_display["Probability (%)"] = (df_display["Probability"] * 100).map("{:.2f}%".format)
                            st.dataframe(
                                df_display[["Class", "Probability (%)"]],
                                hide_index=True,
                                use_container_width=True,
                            )

                    elif response.status_code == 503:
                        st.error("⚠️ Backend model is still loading or unavailable.")
                    elif response.status_code == 400:
                        detail = response.json().get("detail", "Invalid image payload")
                        st.error(f"⚠️ Validation Error: {detail}")
                    else:
                        st.error(f"⚠️ Server Error ({response.status_code}): {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        f"🔴 Connection Failed: Unable to connect to `{api_url}`. "
                        "Please verify the FastAPI server is running."
                    )
                except requests.exceptions.Timeout:
                    st.error(
                        "⏳ Request timed out. If using Render free tier, the server may be waking up from cold sleep. Please retry in 30 seconds."
                    )
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Medical Disclaimer Banner (Mandatory)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Medical Safety & Educational Disclaimer:</strong><br>
        This application is an educational and research machine-learning demonstration. 
        It is <strong>NOT</strong> a certified medical diagnostic device and must <strong>NOT</strong> be used for 
        clinical decision-making, medical diagnosis, or treatment planning. 
        Always consult a qualified healthcare professional for medical diagnoses.
    </div>
    """,
    unsafe_allow_html=True,
)
