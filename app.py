import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from model import predict


st.set_page_config(page_title="Car Brand Identifier", layout="wide", page_icon="🚗")

# ======= HEADER =======
st.markdown("<h1 style='text-align:center;'>🚗 Car Brand Identifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Upload a car photo and the model <a href='https://www.kaggle.com/models/dekxrma/car-brand-identifier'>Car Brand Identifier</a> will predict its brand.</p>", unsafe_allow_html=True)

st.markdown("---")

# ====== LAYOUT (LEFT / RIGHT) ======
left, right = st.columns([0.45, 0.55], gap="large")

with left:
    st.subheader("Input")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible")

    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Center the button
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        run = st.button("🔍 Recognize", use_container_width=True)

with right:
    st.subheader("Prediction")

    if run and image:
        preds = predict(image)

        labels = [p[0] for p in preds]
        probs = [p[1] for p in preds]

        # Pie chart
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            probs,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")

        st.pyplot(fig)

    elif not image:
        st.info("Upload an image and click 'Recognize'.")
