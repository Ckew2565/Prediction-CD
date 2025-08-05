import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Battery Heat Prediction", layout="wide")

# ===== CUSTOM STYLE =====
st.markdown("""
<style>
body { background-color: #f4f9ff; }
.input-label {
    font-size: 22px;
    font-weight: bold;
    margin-top: 16px;
    color: #003366;
}
.header-text {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 8px;
    color: #002244;
}
.output-box {
    background-color: #E6E6FA;
    padding: 16px;
    margin: 10px 0;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    font-size: 20px;
    font-weight: bold;
    color: #222;
}
div.stButton > button {
    width: 100%;
    background-color: #3399CC;  /* สีพื้นหลังปุ่ม */
    color: white;  /* สีตัวอักษร */
    font-size: 80px;  /* ขนาดตัวอักษรในปุ่ม */
    font-weight: bold;
    padding: 15px;
    border: none;
    border-radius: 10px;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #3366CC;  /* สีพื้นหลังเมื่อเลื่อนเมาส์ไปที่ปุ่ม */
    cursor: pointer;  /* เปลี่ยนเคอร์เซอร์เมื่อเมาส์ชี้ที่ปุ่ม */
}
</style>
""", unsafe_allow_html=True)

# ===== LOAD SCALERS =====
BASE_SCALER = r"F:/st/Projectfinall/web/Prediction-CD/Scaler"
scaler1_time = joblib.load(os.path.join(BASE_SCALER, "model1", "model1_scaler_time.pkl"))
scaler1_temp = joblib.load(os.path.join(BASE_SCALER, "model1", "model1_scaler_temp.pkl"))
scaler1_voltage = joblib.load(os.path.join(BASE_SCALER, "model1", "model1_scaler_voltage.pkl"))

scaler2_time = joblib.load(os.path.join(BASE_SCALER, "model2", "model2_scaler_time.pkl"))
scaler2_temp = joblib.load(os.path.join(BASE_SCALER, "model2", "model2_scaler_temp.pkl"))
scaler2_voltage = joblib.load(os.path.join(BASE_SCALER, "model2", "model2_scaler_voltage.pkl"))
scaler2_mean = joblib.load(os.path.join(BASE_SCALER, "model2", "model2_scaler_meantemp.pkl"))
scaler2_max = joblib.load(os.path.join(BASE_SCALER, "model2", "model2_scaler_maxtemp.pkl"))
scaler2_min = joblib.load(os.path.join(BASE_SCALER, "model2", "model2_scaler_mintemp.pkl"))

scaler3_maxtemp = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "model3_scaler_maxtemp.pkl"))
scaler3_meantemp = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "model3_scaler_meantemp.pkl"))
scaler3_mintemp = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "model3_scaler_mintemp.pkl"))
scaler3_temp = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "model3_scaler_temp.pkl"))
scaler3_time = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "model3_scaler_time.pkl"))
scaler3_voltage = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "model3_scaler_voltage.pkl"))
scaler3_Y = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "model3_scaler_Y.pkl"))

# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    base_model = r"F:/st/Projectfinall/web/Prediction-CD/model"
    m2 = load_model(os.path.join(base_model, "model2_2hidden_50epochs.h5"), compile=False)  # Model 1 (Electric Potential)
    m1 = load_model(os.path.join(base_model, "model1_2hidden_50epochs.h5"), compile=False)  # Model 2 (Temperature)
    mh = load_model(os.path.join(base_model, "model3_3hidden_200epochs.h5"), compile=False)  # Model 3 (Heat Flux)
    return m2, m1, mh
model2, model1, model_hf = load_models()

# ===== HEADER =====
cols = st.columns([4, 3, 2, 2])
with cols[1]:
    logos = ["wu.png", "enwu.jpg", "com.png", "ieee.png"]
    logo_cols = st.columns(4)
    for lc, logo, w in zip(logo_cols, logos, [350, 90, 80, 900]):
        path = os.path.join(r"F:/st/Projectfinall/web/Prediction-CD/image", logo)
        if os.path.exists(path):
            with lc: st.image(path, width=w)

st.markdown("<div class='header-text'>Prediction of Heat Distribution in Battery Cells for Electric Vehicles</div>", unsafe_allow_html=True)
st.markdown("<div class='header-text' style='font-size: 32px;'>ระบบคาดการณ์การกระจายความร้อนในแบตเตอรี่สำหรับยานยนต์ไฟฟ้า</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='font-size: 25px;'><b>18650 LiFePO₄ Battery</b></div>", unsafe_allow_html=True)

# ===== LAYOUT (Centered) =====
st.markdown("<div style='padding-left: 40px; padding-right: 40px;'>", unsafe_allow_html=True)
input_col, output_col, plot_col = st.columns([1, 1, 1])

# ===== INPUTS =====
with input_col:
    st.markdown("<div class='input-label'>Select Mode</div>", unsafe_allow_html=True)
    mode = st.radio("", ["Charging", "Discharging"], horizontal=True)

    st.markdown("<div class='input-label'>Battery Operation Time (s)</div>", unsafe_allow_html=True)
    battery_time = st.number_input("", min_value=0.0, value=500.0, step=1.0)

    st.markdown("<div class='input-label'>Ambient Temperature (°C)</div>", unsafe_allow_html=True)
    ambient_temp = st.number_input("", min_value=-50.0, value=40.0, step=0.5)

    if st.button("Predict", use_container_width=True):
        bt, at = float(battery_time), float(ambient_temp)
        flag_c = 1 if mode == "Charging" else 0
        
        # Model 1: Predict Electric Potential
        X1 = np.hstack([ 
            scaler1_time.transform([[bt]]),
            scaler1_temp.transform([[at]]),
            [[flag_c, 1 - flag_c]]  # Charging/Discharging
        ])
        X1 = np.reshape(X1, (1, 4))  # ปรับขนาดให้ตรงกับที่โมเดลคาดหวัง

        out1 = model1.predict(X1)[0]
        ep = scaler1_voltage.inverse_transform([[out1[0]]])[0, 0]  # Electric Potential

        # Model 2: Predict Temperature
        X2 = np.hstack([ 
            scaler2_time.transform([[bt]]),
            scaler2_voltage.transform([[ep]]),  # Electric Potential from Model 1
            scaler2_temp.transform([[at]]),
            [[flag_c, 1 - flag_c]]  # Charging/Discharging
        ])
        X2 = np.reshape(X2, (1, 5))  # ปรับขนาดให้ตรงกับที่โมเดลคาดหวัง

        out2 = model2.predict(X2)[0]
        mean_temp = scaler2_mean.inverse_transform([[out2[0]]])[0, 0]
        max_temp  = scaler2_max.inverse_transform([[out2[1]]])[0, 0]
        min_temp  = scaler2_min.inverse_transform([[out2[2]]])[0, 0]

        # Model 3: Predict Heat Flux
        X3 = np.hstack([ 
            scaler3_time.transform([[bt]]),
            scaler3_temp.transform([[at]]),
            scaler3_voltage.transform([[ep]]),
            scaler3_meantemp.transform([[mean_temp]]),
            scaler3_maxtemp.transform([[max_temp]]),
            scaler3_mintemp.transform([[min_temp]]),
            [[flag_c, 1 - flag_c]]  # Charging/Discharging
        ])
        X3 = np.reshape(X3, (1, 8))  # ปรับขนาดให้ตรงกับที่โมเดลคาดหวัง

        hf_list = model_hf.predict(X3)
        hf_list = scaler3_Y.inverse_transform(hf_list)[0].tolist()

        # Displaying the results with 3 decimal points for Heat Flux
        st.session_state.update({
            'pred': True,
            'ep': ep,
            'mean': mean_temp,
            'max': max_temp,
            'min': min_temp,
            'hf': [round(x, 3) for x in hf_list]  # Round Heat Flux values to 3 decimal places
        })

# ===== DISPLAY RESULTS =====
if st.session_state.get('pred'):
    with output_col:
        st.markdown(f"<div class='output-box'>⚡ Electric Potential: {st.session_state['ep']:.3f} V</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>🌡️ Mean Temp: {st.session_state['mean']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>🔥 Max Temp: {st.session_state['max']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>❄️ Min Temp: {st.session_state['min']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>⚡ Heat Flux: {', '.join([str(i) for i in st.session_state['hf']])}</div>", unsafe_allow_html=True)

    with plot_col:
        # Plot Heat Flux
        xs = np.linspace(0, 65, len(st.session_state['hf']))
        ys = st.session_state['hf']
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
