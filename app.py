from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.predict import load_model, predict_image


st.set_page_config(page_title="Phân loại rác tái chế", page_icon="♻️", layout="centered")

MODEL_PATH = Path("outputs/best_model.pth")

st.title("Phân loại rác thải sinh hoạt")
st.caption("Demo ứng dụng dùng CNN/PyTorch để hỗ trợ nhận diện nhóm rác phục vụ tái chế.")


@st.cache_resource
def get_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size = load_model(model_path, device)
    return model, class_names, image_size, device


if not MODEL_PATH.exists():
    st.warning("Chưa tìm thấy model. Hãy train trước để tạo outputs/best_model.pth.")
    st.code("python src/train.py --data_dir data --epochs 20", language="bash")
    st.stop()

uploaded_file = st.file_uploader("Tải ảnh rác cần phân loại", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    temp_dir = Path("outputs/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / uploaded_file.name
    image.save(temp_path)

    st.image(image, caption="Ảnh đầu vào", use_container_width=True)

    model, class_names, image_size, device = get_model(str(MODEL_PATH))
    predictions = predict_image(model, str(temp_path), class_names, image_size, device, top_k=3)
    top_prediction = predictions[0]

    st.subheader(f"Kết quả: {top_prediction['class_name']}")
    st.progress(top_prediction["probability"])
    st.write(f"Độ tin cậy: **{top_prediction['probability'] * 100:.2f}%**")

    chart_df = pd.DataFrame(predictions)
    chart_df["probability_percent"] = chart_df["probability"] * 100
    st.bar_chart(chart_df.set_index("class_name")["probability_percent"])
