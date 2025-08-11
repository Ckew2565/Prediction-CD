import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Battery Heat Prediction", layout="wide")

# ===== CUSTOM STYLE =====
st.markdown("""
<style>
body { background-color: #f4f9ff; }
.input-label { font-size: 22px; font-weight: bold; margin-top: 16px; color: #003366; }
.header-text { font-size: 40px; font-weight: bold; text-align: center; margin-bottom: 8px; color: #002244; }
.output-box {
    background-color: #E6E6FA; padding: 16px; margin: 10px 0; border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08); font-size: 20px; font-weight: bold; color: #222;
}
/* ปุ่มหลัก (Predict) ให้ใหญ่ */
div.stButton > button {
    width: 100%; background-color: #3399CC; color: white; font-size: 80px; font-weight: bold;
    padding: 15px; border: none; border-radius: 10px; transition: 0.3s;
}
div.stButton > button:hover { background-color: #3366CC; cursor: pointer; }

/* ปุ่มเล็ก (รีเซ็ต) ให้เล็กและอยู่ในแถวเดียวกับหัวข้อ */
.small-icon-btn > button {
    width: auto !important;
    font-size: 18px !important;
    padding: 4px 10px !important;
    background-color: #3399CC !important;
    color: white !important;
    border-radius: 6px !important;
    border: none !important;
}
.small-icon-btn > button:hover {
    background-color: #3366CC !important;
}
</style>
""", unsafe_allow_html=True)

# ===== PATH HELPERS =====
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
SCALER_DIR = ASSETS_DIR / "Scaler"
MODEL_DIR  = ASSETS_DIR / "model"
IMAGE_DIR  = ASSETS_DIR / "image"

def must_exist(p: Path, kind: str = "file") -> Path:
    if (kind == "file" and not p.is_file()) or (kind == "dir" and not p.is_dir()):
        st.error(f"ไม่พบ{'ไฟล์' if kind=='file' else 'โฟลเดอร์'}: {p}")
        st.stop()
    return p

must_exist(ASSETS_DIR, "dir")
must_exist(SCALER_DIR, "dir")
must_exist(MODEL_DIR,  "dir")

# ===== LOAD SCALERS =====
def load_pkl(path: Path):
    try:
        return joblib.load(must_exist(path, "file"))
    except Exception as e:
        st.error(f"โหลดไฟล์ไม่สำเร็จ: {path.name}\nรายละเอียด: {e}")
        st.stop()

scaler1_time    = load_pkl(SCALER_DIR / "model1" / "model1_scaler_time.pkl")
scaler1_temp    = load_pkl(SCALER_DIR / "model1" / "model1_scaler_temp.pkl")
scaler1_voltage = load_pkl(SCALER_DIR / "model1" / "model1_scaler_voltage.pkl")

scaler2_time    = load_pkl(SCALER_DIR / "model2" / "model2_scaler_time.pkl")
scaler2_temp    = load_pkl(SCALER_DIR / "model2" / "model2_scaler_temp.pkl")
scaler2_voltage = load_pkl(SCALER_DIR / "model2" / "model2_scaler_voltage.pkl")
scaler2_mean    = load_pkl(SCALER_DIR / "model2" / "model2_scaler_meantemp.pkl")
scaler2_max     = load_pkl(SCALER_DIR / "model2" / "model2_scaler_maxtemp.pkl")
scaler2_min     = load_pkl(SCALER_DIR / "model2" / "model2_scaler_mintemp.pkl")

scaler3_maxtemp  = load_pkl(SCALER_DIR / "model_heatflux" / "model3_scaler_maxtemp.pkl")
scaler3_meantemp = load_pkl(SCALER_DIR / "model_heatflux" / "model3_scaler_meantemp.pkl")
scaler3_mintemp  = load_pkl(SCALER_DIR / "model_heatflux" / "model3_scaler_mintemp.pkl")
scaler3_temp     = load_pkl(SCALER_DIR / "model_heatflux" / "model3_scaler_temp.pkl")
scaler3_time     = load_pkl(SCALER_DIR / "model_heatflux" / "model3_scaler_time.pkl")
scaler3_voltage  = load_pkl(SCALER_DIR / "model_heatflux" / "model3_scaler_voltage.pkl")
scaler3_Y        = load_pkl(SCALER_DIR / "model_heatflux" / "model3_scaler_Y.pkl")

# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    m2 = load_model(must_exist(MODEL_DIR / "model2_2hidden_50epochs.h5"), compile=False)
    m1 = load_model(must_exist(MODEL_DIR / "model1_2hidden_50epochs.h5"), compile=False)
    mh = load_model(must_exist(MODEL_DIR / "model3_3hidden_200epochs.h5"), compile=False)
    return m2, m1, mh

model2, model1, model_hf = load_models()

# ===== HEADER =====
cols = st.columns([4, 3, 2, 2])
with cols[1]:
    logos = ["wu.png", "enwu.jpg", "com.png", "ieee.png"]
    logo_cols = st.columns(4)
    for lc, logo, w in zip(logo_cols, logos, [350, 90, 80, 900]):
        p = IMAGE_DIR / logo
        if p.exists():
            with lc:
                st.image(str(p), width=w)

st.markdown("<div class='header-text'>Prediction of Heat Distribution in Battery Cells for Electric Vehicles</div>", unsafe_allow_html=True)
st.markdown("<div class='header-text' style='font-size: 32px;'>ระบบคาดการณ์การกระจายความร้อนในแบตเตอรี่สำหรับยานยนต์ไฟฟ้า</div>", unsafe_allow_html=True)

st.markdown("---")

# ===== TITLE + ปุ่มรีเซ็ตเล็ก =====
col_title, col_btn = st.columns([8, 1])
with col_title:
    st.markdown("<div style='font-size: 25px;'><b>18650 LiFePO₄ Battery</b></div>", unsafe_allow_html=True)
with col_btn:
    st.markdown("<div class='small-icon-btn'>", unsafe_allow_html=True)
    if st.button("🔄", help="ทำนายใหม่"):
        for k in ["pred", "ep", "mean", "max", "min", "hf", "bt_input", "at_input", "mode_radio"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ===== LAYOUT =====
st.markdown("<div style='padding-left: 40px; padding-right: 40px;'>", unsafe_allow_html=True)
input_col, output_col, plot_col = st.columns([1, 1, 1])

if "pred" not in st.session_state:
    st.session_state["pred"] = False

# ===== INPUTS =====
with input_col:
    st.markdown("<div class='input-label'>Select Mode</div>", unsafe_allow_html=True)
    mode = st.radio("", ["Charging", "Discharging"], horizontal=True, key="mode_radio")

    st.markdown("<div class='input-label'>Battery Operation Time (s)</div>", unsafe_allow_html=True)
    battery_time_str = st.text_input("", value=st.session_state.get("bt_input", ""), key="bt_input",
                                     placeholder="Enter operation time in seconds")

    st.markdown("<div class='input-label'>Ambient Temperature (°C)</div>", unsafe_allow_html=True)
    ambient_temp_str = st.text_input("", value=st.session_state.get("at_input", ""), key="at_input",
                                     placeholder="Enter ambient temperature in °C")

    if st.button("Predict", use_container_width=True):
        if battery_time_str.strip() == "" or ambient_temp_str.strip() == "":
            st.warning("⚠️ กรุณากรอกข้อมูลให้ครบทั้งสองช่อง")
            st.stop()
        try:
            bt = float(battery_time_str)
            at = float(ambient_temp_str)
        except ValueError:
            st.error("❌ กรุณากรอกตัวเลขเท่านั้น")
            st.stop()

        flag_c = 1 if st.session_state["mode_radio"] == "Charging" else 0

        # Model 1
        X1 = np.hstack([
            scaler1_time.transform([[bt]]),
            scaler1_temp.transform([[at]]),
            [[flag_c, 1 - flag_c]]
        ]).reshape(1, 4)
        out1 = model1.predict(X1, verbose=0)[0]
        ep = scaler1_voltage.inverse_transform([[out1[0]]])[0, 0]

        # Model 2
        X2 = np.hstack([
            scaler2_time.transform([[bt]]),
            scaler2_voltage.transform([[ep]]),
            scaler2_temp.transform([[at]]),
            [[flag_c, 1 - flag_c]]
        ]).reshape(1, 5)
        out2 = model2.predict(X2, verbose=0)[0]
        mean_temp = scaler2_mean.inverse_transform([[out2[0]]])[0, 0]
        max_temp  = scaler2_max.inverse_transform([[out2[1]]])[0, 0]
        min_temp  = scaler2_min.inverse_transform([[out2[2]]])[0, 0]

        # Model 3
        X3 = np.hstack([
            scaler3_time.transform([[bt]]),
            scaler3_temp.transform([[at]]),
            scaler3_voltage.transform([[ep]]),
            scaler3_meantemp.transform([[mean_temp]]),
            scaler3_maxtemp.transform([[max_temp]]),
            scaler3_mintemp.transform([[min_temp]]),
            [[flag_c, 1 - flag_c]]
        ]).reshape(1, 8)
        hf_list = model_hf.predict(X3, verbose=0)
        hf_list = scaler3_Y.inverse_transform(hf_list)[0].tolist()

        st.session_state.update({
            "pred": True,
            "ep": ep,
            "mean": mean_temp,
            "max": max_temp,
            "min": min_temp,
            "hf": [round(x, 3) for x in hf_list]
        })

# ===== DISPLAY RESULTS =====
if st.session_state.get("pred"):
    with output_col:
        st.markdown(f"<div class='output-box'>⚡ Electric Potential: {st.session_state['ep']:.3f} V</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>🌡️ Mean Temp: {st.session_state['mean']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>🔥 Max Temp: {st.session_state['max']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>❄️ Min Temp: {st.session_state['min']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>⚡ Heat Flux: {', '.join([str(i) for i in st.session_state['hf']])}</div>", unsafe_allow_html=True)

    with plot_col:
        xs = np.linspace(0, 65, len(st.session_state["hf"]))
        ys = st.session_state["hf"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker='o', linestyle='-', color='#1f77b4',
                markerfacecolor='orange', markersize=6)
        ax.set_xlim(0, 65)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 65])
        ax.set_xlabel('Arc Length (mm)', fontsize=10)
        ax.set_ylabel('Heat Flux (W/m²)', fontsize=10)
        ax.set_title('Graph of Heat Flux Distribution Along Arc Length', fontsize=12, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_facecolor('#f7f7f7')
        st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
