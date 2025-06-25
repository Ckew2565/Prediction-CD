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
    background-color: #9999FF;
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 12px;
    border: none;
    border-radius: 10px;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #99CCFF;
}
input[type="number"] {
    font-size: 20px !important;
    padding: 10px !important;
    height: 50px !important;
    width: 100% !important;
    border-radius: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# ===== LOAD SCALERS =====
BASE_SCALER = r"F:/st/Projectfinall/web/Prediction-CD/Scaler"
scaler1_time    = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_time.pkl"))
scaler1_temp    = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler1_temp.pkl"))
scaler1_Y       = joblib.load(os.path.join(BASE_SCALER, "model1", "scaler2_Y.pkl"))

scaler2_time    = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_time.pkl"))
scaler2_temp    = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_temp.pkl"))
scaler2_voltage = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_voltage.pkl"))
scaler2_current = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_current.pkl"))
scaler2_mean    = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_meantemp.pkl"))
scaler2_max     = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_maxtemp.pkl"))
scaler2_min     = joblib.load(os.path.join(BASE_SCALER, "model2", "scaler2_mintemp.pkl"))

scaler3_X       = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "scaler3_X.pkl"))
scaler3_Y       = joblib.load(os.path.join(BASE_SCALER, "model_heatflux", "scaler3_Y.pkl"))

# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    base_model = r"F:/st/Projectfinall/web/Prediction-CD/model" 
    model_epcc = load_model(os.path.join(base_model, "model1_2h_32bs.h5"), compile=False)
    model_temp = load_model(os.path.join(base_model, "ann_model_temperature.h5"), compile=False)
    model_hf   = load_model(os.path.join(base_model, "model3_6h_16bs.h5"), compile=False)
    return model_epcc, model_temp, model_hf
model_epcc, model_temp, model_hf = load_models()

# ===== HEADER =====
cols = st.columns([4, 3, 4])
with cols[1]:
    logos = ["wu.png", "enwu.jpg", "com.png"]
    logo_cols = st.columns(3)
    for lc, logo, w in zip(logo_cols, logos, [55, 90, 80]):
        path = os.path.join(r"F:/st/Projectfinall/web/Prediction-CD/image", logo)
        if os.path.exists(path):
            with lc: st.image(path, width=w)

st.markdown("<div class='header-text'>Prediction of Heat Distribution in Battery Cells for Electric Vehicles</div>", unsafe_allow_html=True)
st.markdown("---")

# ===== LAYOUT =====
st.markdown("<div style='padding-left: 40px; padding-right: 40px;'>", unsafe_allow_html=True)
input_col, output_col, plot_col = st.columns([1, 1, 1])

# ===== INPUT SECTION =====
with input_col:
    st.markdown("<div class='input-label'>Battery Operation Time (s)</div>", unsafe_allow_html=True)
    battery_time = st.number_input("", min_value=0.0, value=500.0, step=1.0)

    st.markdown("<div class='input-label'>Ambient Temperature (°C)</div>", unsafe_allow_html=True)
    ambient_temp = st.number_input("", min_value=-50.0, value=40.0, step=0.5)

    if st.button("Predict", use_container_width=True):
        bt, at = float(battery_time), float(ambient_temp)

        # ----- MODEL 1: Predict ep, cc -----
        X1 = np.hstack([
            scaler1_time.transform([[bt]]),
            scaler1_temp.transform([[at]])
        ])
        out1 = model_epcc.predict(X1)[0]
        ep, cc = scaler1_Y.inverse_transform(out1.reshape(1, -1))[0]

        # ===== Calculate charge/discharge =====
        if cc > 0:
            charge, discharge = 1, 0
        elif cc < 0:
            charge, discharge = 0, 1
        else:
            charge, discharge = 0, 0

        # ----- MODEL 2: Predict mean/max/min temp -----
        X2 = np.hstack([
            scaler2_time.transform([[bt]]),
            scaler2_temp.transform([[at]]),
            scaler2_voltage.transform([[ep]]),
            scaler2_current.transform([[cc]]),
            [[charge]],
            [[discharge]]
        ])
        out2 = model_temp.predict(X2)[0]
        mean_temp = scaler2_mean.inverse_transform([[out2[0]]])[0, 0]
        max_temp  = scaler2_max.inverse_transform([[out2[1]]])[0, 0]
        min_temp  = scaler2_min.inverse_transform([[out2[2]]])[0, 0]

        # ----- MODEL 3: Predict Heat Flux -----
        hf_list = scaler3_Y.inverse_transform(
            model_hf.predict(scaler3_X.transform([[bt, at, ep]]))
        )[0].tolist()

        st.session_state.update({
            'pred': True,
            'mean': mean_temp,
            'max': max_temp,
            'min': min_temp,
            'ep': ep,
            'cc': cc,
            'hf': hf_list
        })

# ===== DISPLAY RESULTS =====
if st.session_state.get('pred'):
    with output_col:
        st.markdown(f"<div class='output-box'>🌡️ Mean Temp: {st.session_state['mean']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>🔥 Max Temp: {st.session_state['max']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>❄️ Min Temp: {st.session_state['min']:.2f} °C</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>⚡ Electric Potential: {st.session_state['ep']:.2f} V</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='output-box'>🔋 Cell Current: {st.session_state['cc']:.2f} A</div>", unsafe_allow_html=True)
        #st.markdown(f"<div class='output-box'>🔌 Charge (Input to model): {charge}</div>", unsafe_allow_html=True)
        #st.markdown(f"<div class='output-box'>🔋 Discharge (Input to model): {discharge}</div>", unsafe_allow_html=True)

        

    with plot_col:
        xs = np.linspace(0, 65, len(st.session_state['hf']))
        ys = st.session_state['hf']
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(xs, ys, marker='o', linestyle='-', color='#1f77b4',
                markerfacecolor='orange', markersize=6)
        ax.set_xlim(0, 65)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 65])
        ax.set_xlabel('Cell Length (mm)', fontsize=10)
        ax.set_ylabel('Heat Flux (W/m²)', fontsize=10)
        ax.set_title('Heat Flux Distribution', fontsize=12, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_facecolor('#f7f7f7')
        st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
