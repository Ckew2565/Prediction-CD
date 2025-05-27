import streamlit as st
from PIL import Image

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Battery Heat Prediction", layout="wide")

# ========== STYLE ==========
st.markdown("""
    <style>
    .input-label {
        font-size: 30px;
        font-weight: bold;
        margin-top: 10px;
    }
    .header-text {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        padding-top: 20px;
    }
    .big-radio .stRadio > div > label {
        font-size: 50px !important;
        padding: 10px 20px;
    }
    .big-radio .stRadio > div {
        gap: 50px;
    }
    .big-radio input[type="radio"] {
        width: 40px;
        height: 40px;
    }
    div.stButton > button {
        width: 100%;
        background-color: #ADD8E6;
        color: black;
        font-size: 24px;
        font-weight: bold;
        padding: 15px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #87CEEB;
    }
    .temperature-box {
        font-size: 40px;
        font-weight: bold;
        color: black;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        width: 100%;
        box-sizing: border-box;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ========== HEADER SECTION ==========
col_left, col_center, col_right = st.columns([4, 6, 4])

with col_center:
    # แบ่งโลโก้ 6 รูปในแถวเดียวกัน (ชิดกัน)
    logo1, logo2, logo3, logo4, logo5, logo6 = st.columns(6)

    with logo1:
        st.image("F:/st/Projectfinall/web/Prediction-CD/image/LogoMHES.png", width=80)

    with logo2:
        st.image("F:/st/Projectfinall/web/Prediction-CD/image/Logo NRCT.png", width=60)

    with logo3:
        st.image("F:/st/Projectfinall/web/Prediction-CD/image/Logo Expo2025.png", width=120)

    with logo4:
        st.image("F:/st/Projectfinall/web/Prediction-CD/image/wu.png", width=50)

    with logo5:
        st.image("F:/st/Projectfinall/web/Prediction-CD/image/enwu.jpg", width=90)

    with logo6:
        st.image("F:/st/Projectfinall/web/Prediction-CD/image/com.png", width=80)


# ชั้นที่ 2: หัวข้ออยู่ตรงกลางใต้โลโก้
st.markdown("""
    <div class='header-text'>
        Prediction of Heat Distribution in Battery Cells for Electric Vehicles
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ========== MAIN LAYOUT SECTION (2 Columns) ==========
left_col, right_col = st.columns(2)

# ---------- LEFT COLUMN: MODE + INPUT ---------- #
with left_col:
    st.markdown("<div style='font-size:40px; font-weight:bold;'>Select Mode</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:30px; font-weight:bold; margin-top:10px;'>Battery Operation Mode:</div>", unsafe_allow_html=True)

    # Big Radio Button
    with st.container():
     st.markdown('<div class="big-radio">', unsafe_allow_html=True)
    mode = st.radio(
        label="",
        options=["Charging", "Discharging"],
        index=1,
        horizontal=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


    # Input Fields
    st.markdown("<div class='input-label'>Battery Operation Time (S)</div>", unsafe_allow_html=True)
    battery_time = st.text_input(" ", value="500")

    st.markdown("<div class='input-label'>Ambient Temperature (°C)</div>", unsafe_allow_html=True)
    ambient_temp = st.text_input("  ", value="40")

# ---------- RIGHT COLUMN: OUTPUT + PREDICT BUTTON ---------- #
with right_col:
    st.markdown("<br>", unsafe_allow_html=True)

    # --------- เริ่มเซต Session State ---------
    if "predicted_temp" not in st.session_state:
        st.session_state.predicted_temp = 0.0  # เริ่มต้นที่ 0 องศา
    if "predicted" not in st.session_state:
        st.session_state.predicted = False  # ยังไม่ได้กด Predict
    # --------- จบเซต Session State ---------

    # ถ้ากดทำนายแล้ว ค่อยโชว์กราฟ
    if st.session_state.predicted:
        image = Image.open("F:/st/Projectfinall/web/Prediction-CD/image/graph.jpg")
        graph_width = 600
        st.image(image, caption="Predicted Heat Map", width=graph_width)
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        # ถ้ายังไม่กดทำนาย ก็เว้นที่กราฟไว้เฉยๆ (กันกระพริบ)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # กล่องโชว์อุณหภูมิอยู่ที่เดิมตลอด
    st.markdown(f"<div class='temperature-box'>{st.session_state.predicted_temp:.2f} °C</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ปุ่ม Predict
    if st.button("predict", use_container_width=True):
        # แปลงค่าจาก input (ถ้าไม่กรอกให้เป็น 0)
        battery_time_value = float(battery_time) if battery_time.strip() else 0.0
        ambient_temp_value = float(ambient_temp) if ambient_temp.strip() else 0.0

        # คำนวณอุณหภูมิใหม่
        new_predicted_temp = 30 + 0.05 * battery_time_value + 0.3 * ambient_temp_value

        # อัปเดต Session State
        st.session_state.predicted_temp = new_predicted_temp
        st.session_state.predicted = True  # เปิด flag ว่าเคยกด predict แล้ว

        st.success(f"Prediction complete for mode: {mode}")

