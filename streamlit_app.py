import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Battery Heat Prediction", layout="wide")

# ===== LOAD SCALERS =====
BASE_SCALER = r"F:/st/Projectfinall/web/Prediction-CD/Scaler"

# Model2 scalers for EP and CC
scaler2_time    = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_time.pkl"))
scaler2_temp    = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_temp.pkl"))
scaler2_voltage = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_voltage.pkl"))
scaler2_current = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_current.pkl"))

# Model1 scalers for features
scaler1_time    = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_time.pkl"))
scaler1_temp    = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_temp.pkl"))
scaler1_voltage = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_voltage.pkl"))
scaler1_current = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_current.pkl"))
# Model1 scaler for outputs (mean, max, min)
scaler1_Y       = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler2_Y.pkl"))

# Model_hf scalers for heat flux
scaler3_X = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "scaler3_X.pkl"))
scaler3_Y = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "scaler3_Y.pkl"))

# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    base_model = r"F:/st/Projectfinall/web/Prediction-CD/model"
    model2 = load_model(os.path.join(base_model, "model1_2h_32bs.h5"), compile=False)
    model1 = load_model(os.path.join(base_model, "ann_model_temperature.h5"), compile=False)
    model_hf = load_model(os.path.join(base_model, "model_heatflux.h5"), compile=False)
    return model2, model1, model_hf

model2, model1, model_hf = load_models()

# ===== STYLE =====
st.markdown("""
<style>
.input-label { font-size: 24px; font-weight: bold; margin-top: 16px; }
.header-text { font-size: 32px; font-weight: bold; text-align: center; margin-bottom: 8px; }
.output-box { font-size: 24px; font-weight: bold; margin-bottom: 8px; }
.hf-box { font-size: 16px; font-weight: bold; border: 1px solid #ccc; border-radius: 6px; padding: 8px; text-align: center; margin: 4px; }
div.stButton > button { width: 100%; background-color: #ADD8E6; color: black; font-size: 20px; font-weight: bold; padding: 12px; border: none; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
cols = st.columns([2,6,2])
with cols[1]:
    logo_cols = st.columns(6)
    logos = ["LogoMHES.png", "Logo NRCT.png", "Logo Expo2025.png", "wu.png", "enwu.jpg", "com.png"]
    widths = [80,60,120,50,90,80]
    for lc, logo, w in zip(logo_cols, logos, widths):
        path = os.path.join(r"F:/st/Projectfinall/web/Prediction-CD/image", logo)
        if os.path.exists(path):
            with lc:
                st.image(path, width=w)

st.markdown("<div class='header-text'>Prediction of Heat Distribution in Battery Cells for Electric Vehicles</div>", unsafe_allow_html=True)
st.markdown("---")

# ===== LAYOUT =====
input_col, output_col, plot_col = st.columns([2,3,3])

# -- INPUTS --
with input_col:
    st.markdown("<div class='input-label'>Select Mode</div>", unsafe_allow_html=True)
    mode = st.radio("", ["Charging", "Discharging"], horizontal=True)
    st.markdown("<div class='input-label'>Battery Operation Time (s)</div>", unsafe_allow_html=True)
    battery_time = st.number_input("", min_value=0.0, value=500.0, step=1.0)
    st.markdown("<div class='input-label'>Ambient Temperature (°C)</div>", unsafe_allow_html=True)
    ambient_temp = st.number_input("", min_value=-50.0, value=40.0, step=0.5)
    predict_btn = st.button("Predict")

# -- PREDICT LOGIC --
if predict_btn:
    bt = float(battery_time)
    at = float(ambient_temp)
    charge_flag = 1 if mode == "Charging" else 0
    discharge_flag = 1 if mode == "Discharging" else 0

    # Model2 → EP, CC
    raw2 = np.array([[bt, at, charge_flag, discharge_flag]], dtype=np.float32)
    t2 = scaler2_time.transform(raw2[:, [0]])
    temp2 = scaler2_temp.transform(raw2[:, [1]])
    flags2 = raw2[:, 2:].reshape(1,2)
    X2 = np.hstack([t2, temp2, flags2])
    y2 = model2.predict(X2)[0].reshape(1,-1)
    ep_val = float(scaler2_voltage.inverse_transform(y2[:, [0]])[0,0])
    cc_val = float(scaler2_current.inverse_transform(y2[:, [1]])[0,0])

    # Model1 → Mean/Max/Min
    raw1 = np.array([[bt, ep_val, cc_val, at, charge_flag, discharge_flag]], dtype=np.float32)
    t1 = scaler1_time.transform(raw1[:, [0]])
    ep1 = scaler1_voltage.transform(raw1[:, [1]])
    cc1 = scaler1_current.transform(raw1[:, [2]])
    temp1 = scaler1_temp.transform(raw1[:, [3]])
    flags1 = raw1[:, 4:].reshape(1,2)
    X1 = np.hstack([t1, ep1, cc1, temp1, flags1])
    y1 = model1.predict(X1)[0].reshape(1,-1)
    m, M, m_ = scaler1_Y.inverse_transform(y1)[0]

    # Model_hf → Heat Flux
    raw3 = np.array([[bt, at, charge_flag]], dtype=np.float32)
    X3 = scaler3_X.transform(raw3)
    y3 = model_hf.predict(X3)[0].reshape(1,-1)
    hf_list = scaler3_Y.inverse_transform(y3)[0].tolist()

    # store in session_state
    st.session_state['pred'] = True
    st.session_state['mean_temp'] = m
    st.session_state['max_temp'] = M
    st.session_state['min_temp'] = m_
    st.session_state['ep'] = ep_val
    st.session_state['cc'] = cc_val
    st.session_state['hf_list'] = hf_list

# -- DISPLAY RESULTS --
if st.session_state.get('pred', False):
    with output_col:
        st.markdown(f"<div class='output-box'>Mean Temp: {st.session_state['mean_temp']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>Max Temp: {st.session_state['max_temp']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>Min Temp: {st.session_state['min_temp']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>Electric Potential (V): {st.session_state['ep']:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>Cell Current (A): {st.session_state['cc']:.2f}</div>", unsafe_allow_html=True)
        #st.markdown("<div class='input-label'>Heat Flux Predictions</div>", unsafe_allow_html=True)
       # hf_cols = st.columns(5)
        #for i, val in enumerate(st.session_state['hf_list']):
           # with hf_cols[i % 5]:
                #st.markdown(f"<div class='hf-box'>HF {i+1}: {val:.2f}</div>", unsafe_allow_html=True)

    with plot_col:
        xs = np.linspace(0, 65, len(st.session_state['hf_list']))
        ys = st.session_state['hf_list']
        # ปรับขนาดกราฟให้เล็กลง
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(xs, ys, marker='o', linestyle='-')
        ax.set_xlabel('Cell length (mm)')
        ax.set_ylabel('Heat flux (W/m²)')
        ax.set_title('Heat Flux Distribution Along Battery Cell')
        ax.grid(True)
        st.pyplot(fig)
