import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Battery Heat Prediction", layout="wide")

# ========== โหลด SCALER ด้วย joblib ==========
BASE_SCALER = r"F:/st/Projectfinall/web/Prediction-CD/Scaler"

# --- model2 scalers (EP, CC) ---
scaler2_time    = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_time.pkl"))
scaler2_temp    = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_temp.pkl"))
scaler2_voltage = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_voltage.pkl"))
scaler2_current = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_current.pkl"))

# --- model1 scalers (features) ---
scaler1_time    = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_time.pkl"))
scaler1_temp    = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_temp.pkl"))
scaler1_voltage = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_voltage.pkl"))
scaler1_current = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_current.pkl"))
# --- model1 scaler สำหรับ target 3 ค่า (Mean/Max/Min) ---
scaler1_Y       = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler2_Y.pkl"))

# --- model_heatflux scalers (ใช้สเกล Xhf → inverse Yhf) ---
scaler3_X       = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "scaler3_X.pkl"))
scaler3_Y       = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "scaler3_Y.pkl"))

# ========== LOAD MODELS ==========
@st.cache_resource
def load_models():
    base_model = r"F:/st/Projectfinall/web/Prediction-CD/model"
    model2   = load_model(os.path.join(base_model, "model1_2h_32bs.h5"), compile=False)
    model1   = load_model(os.path.join(base_model, "ann_model_temperature.h5"), compile=False)
    model_hf = load_model(os.path.join(base_model, "model_heatflux.h5"), compile=False)
    return model2, model1, model_hf

model2, model1, model_hf = load_models()

# ========== STYLE ==========
st.markdown("""
    <style>
    .input-label { font-size: 28px; font-weight: bold; margin-top: 12px; }
    .header-text { font-size: 36px; font-weight: bold; text-align: center; padding-top: 15px; }
    .output-box { font-size: 28px; font-weight: bold; color: black; padding: 12px; margin-top: 12px; }
    .hf-box { font-size: 18px; font-weight: bold; color: black; border: 1px solid #ccc; border-radius: 6px; padding: 8px; text-align: center; margin: 4px; }
    div.stButton > button { width: 100%; background-color: #ADD8E6; color: black; font-size: 20px; font-weight: bold; padding: 12px; border: none; border-radius: 8px; cursor: pointer; }
    div.stButton > button:hover { background-color: #87CEEB; }
    </style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
col1, col2, col3 = st.columns([2,6,2])
with col2:
    logo_cols = st.columns(6)
    logos = ["LogoMHES.png", "Logo NRCT.png", "Logo Expo2025.png", "wu.png", "enwu.jpg", "com.png"]
    widths = [80,60,120,50,90,80]
    for lc, logo, w in zip(logo_cols, logos, widths):
        img_path = os.path.join(r"F:/st/Projectfinall/web/Prediction-CD/image", logo)
        if os.path.exists(img_path):
            with lc:
                st.image(img_path, width=w)

st.markdown("""
<div class='header-text'>
    Prediction of Heat Distribution in Battery Cells for Electric Vehicles
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ========== INPUTS & PREDICT BUTTON ==========
left_col, right_col = st.columns(2)
with left_col:
    st.markdown("<div class='input-label'>Select Mode</div>", unsafe_allow_html=True)
    mode = st.radio("", ["Charging", "Discharging"], index=0, horizontal=True)

    st.markdown("<div class='input-label'>Battery Operation Time (s)</div>", unsafe_allow_html=True)
    battery_time = st.number_input("", min_value=0.0, value=500.0, step=1.0, key="battery_time")

    st.markdown("<div class='input-label'>Ambient Temperature (°C)</div>", unsafe_allow_html=True)
    ambient_temp = st.number_input("", min_value=-50.0, value=40.0, step=0.5, key="ambient_temp")

with right_col:
    if st.button("Predict", use_container_width=True):
        # ─── แปลงค่า Input ให้เป็น float ───
        bt = float(battery_time)
        at = float(ambient_temp)
        charge_flag    = 1 if mode == "Charging" else 0
        discharge_flag = 1 if mode == "Discharging" else 0

        # ─── STEP 1: Model2 → EP, CC ───
        raw_X2 = np.array([[bt, at, charge_flag, discharge_flag]], dtype=np.float32)  # (1,4)
        X2_time_s = scaler2_time.transform(raw_X2[:, [0]])    # (1,1)
        X2_temp_s = scaler2_temp.transform(raw_X2[:, [1]])    # (1,1)
        X2_flags  = raw_X2[:, 2:].reshape(1,2)                # (1,2)
        X2_scaled = np.hstack([X2_time_s, X2_temp_s, X2_flags])  # (1,4)

        Y2_scaled_pred = model2.predict(X2_scaled)[0].reshape(1,-1)  # (1,2)
        ep_val = float(scaler2_voltage.inverse_transform(Y2_scaled_pred[:, [0]])[0,0])
        cc_val = float(scaler2_current.inverse_transform(Y2_scaled_pred[:, [1]])[0,0])

        # ─── STEP 2: Model1 → Mean/Max/Min Temp ───
        raw_X1 = np.array([[bt, ep_val, cc_val, at, charge_flag, discharge_flag]], dtype=np.float32)  # (1,6)
        # สเกลทีละฟีเจอร์
        col1_time_s  = scaler1_time.transform(raw_X1[:, [0]])    # (1,1)
        col1_ep_s    = scaler1_voltage.transform(raw_X1[:, [1]]) # (1,1)
        col1_cc_s    = scaler1_current.transform(raw_X1[:, [2]]) # (1,1)
        col1_at_s    = scaler1_temp.transform(raw_X1[:, [3]])    # (1,1)
        col1_flags   = raw_X1[:, 4:].reshape(1,2)                # (1,2)
        X1_scaled    = np.hstack([col1_time_s, col1_ep_s, col1_cc_s, col1_at_s, col1_flags])  # (1,6)

        Y1_scaled_pred = model1.predict(X1_scaled)[0].reshape(1,-1)  # (1,3)
        # inverse‐transform ทั้ง 3 ค่าในคราวเดียว
        Y1_real = scaler1_Y.inverse_transform(Y1_scaled_pred)  # (1,3)
        mean_temp = float(Y1_real[0, 0])
        max_temp  = float(Y1_real[0, 1])
        min_temp  = float(Y1_real[0, 2])

        # ─── STEP 3: Model_hf → Heat Flux 15 ค่า ───
        raw_Xhf        = np.array([[bt, at, charge_flag]], dtype=np.float32)  # (1,3)
        Xhf_scaled     = scaler3_X.transform(raw_Xhf)                         # (1,3)
        Yhf_scaled_pred= model_hf.predict(Xhf_scaled)[0].reshape(1, -1)        # (1,15)
        hf_list        = scaler3_Y.inverse_transform(Yhf_scaled_pred)[0].tolist()  # length=15

        # เก็บผลลัพธ์ลง session_state
        st.session_state['predicted']   = True
        st.session_state['mean_temp']   = mean_temp
        st.session_state['max_temp']    = max_temp
        st.session_state['min_temp']    = min_temp
        st.session_state['ep']          = ep_val
        st.session_state['cc']          = cc_val
        st.session_state['hf_list']     = hf_list

    # ─── แสดงผลลัพธ์ ───
    if st.session_state.get('predicted', False):
        st.markdown(f"<div class='output-box'>Mean Temperature: {st.session_state['mean_temp']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>Max Temperature: {st.session_state['max_temp']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>Min Temperature: {st.session_state['min_temp']:.2f} °C</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='output-box'>Electric Potential (V): {st.session_state['ep']:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>Cell Current (A): {st.session_state['cc']:.2f}</div>", unsafe_allow_html=True)

        st.markdown("<div class='input-label'>Heat Flux Predictions</div>", unsafe_allow_html=True)
        hf_cols = st.columns(5)
        for i, hf_val in enumerate(st.session_state['hf_list']):
            with hf_cols[i % 5]:
                st.markdown(f"<div class='hf-box'>HF {i+1}: {hf_val:.2f}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='output-box'>No prediction yet</div>", unsafe_allow_html=True)
