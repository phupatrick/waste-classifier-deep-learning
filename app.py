from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from src.predict import load_model, predict_image


st.set_page_config(page_title="Phân loại rác tái chế", page_icon="♻️", layout="centered")

MODEL_PATH = Path("outputs/best_model.pth")
LOW_CONFIDENCE_THRESHOLD = 0.5

CLASS_LABELS_VI = {
    "cardboard": "bìa carton",
    "glass": "thủy tinh",
    "metal": "kim loại",
    "paper": "giấy",
    "plastic": "nhựa",
    "trash": "rác khác",
}

PLASTIC_IMAGENET_HINTS = {
    "water bottle",
    "pop bottle",
    "plastic bag",
    "bottlecap",
    "packet",
}


def to_vi_label(class_name: str) -> str:
    return CLASS_LABELS_VI.get(class_name, class_name)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 820px;
        padding-top: 1.5rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
    h1 {
        font-size: 2rem !important;
        line-height: 1.2 !important;
    }
    h2, h3 {
        font-size: 1.25rem !important;
        line-height: 1.3 !important;
    }
    .result-box {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px 18px;
        margin: 12px 0 14px 0;
        background: #f8fafc;
    }
    .result-label {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .result-value {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.25;
    }
    .confidence-note {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_waste_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size = load_model(model_path, device)
    return model, class_names, image_size, device


@st.cache_resource
def get_auxiliary_imagenet_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()
    return model, weights, weights.transforms()


@torch.no_grad()
def has_plastic_object_hint(image: Image.Image) -> bool:
    """Auxiliary check for out-of-distribution images containing many bottles."""
    model, weights, transform = get_auxiliary_imagenet_model()
    tensor = transform(image).unsqueeze(0)
    probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    _, indices = torch.topk(probabilities, k=10)
    labels = {weights.meta["categories"][idx.item()] for idx in indices}
    return bool(labels.intersection(PLASTIC_IMAGENET_HINTS))


def apply_low_confidence_postprocess(predictions, image: Image.Image):
    top_prediction = predictions[0]
    if top_prediction["probability"] >= LOW_CONFIDENCE_THRESHOLD:
        return predictions, False

    if top_prediction["class_name"] == "plastic":
        return predictions, False

    if not has_plastic_object_hint(image):
        return predictions, False

    adjusted = []
    for item in predictions:
        adjusted.append(dict(item))

    class_names = {item["class_name"] for item in adjusted}
    if "plastic" not in class_names:
        adjusted.append({"class_name": "plastic", "probability": 0.0})

    for item in adjusted:
        if item["class_name"] == "plastic":
            item["probability"] = max(item["probability"], 0.56)

    total = sum(item["probability"] for item in adjusted)
    for item in adjusted:
        item["probability"] = item["probability"] / total

    adjusted = sorted(adjusted, key=lambda item: item["probability"], reverse=True)
    return adjusted, True


st.title("Phân loại rác thải sinh hoạt")
st.caption("Ứng dụng demo sử dụng MobileNetV2/PyTorch để hỗ trợ nhận diện nhóm rác phục vụ tái chế.")

if not MODEL_PATH.exists():
    st.warning("Chưa tìm thấy mô hình đã huấn luyện. Hãy chạy lệnh train trước để tạo file outputs/best_model.pth.")
    st.code(
        "python src/train.py --data_dir data --epochs 10 --batch_size 32 --image_size 160 "
        "--architecture mobilenet_v2 --pretrained --class_weights --lr 0.001",
        language="bash",
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Tải ảnh rác cần phân loại",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    temp_dir = Path("outputs/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / uploaded_file.name
    image.save(temp_path)

    st.image(image, caption="Ảnh đầu vào", use_container_width=True)

    model, class_names, image_size, device = get_waste_model(str(MODEL_PATH))
    predictions = predict_image(model, str(temp_path), class_names, image_size, device, top_k=6)
    predictions, postprocessed = apply_low_confidence_postprocess(predictions, image)
    top_prediction = predictions[0]

    result_title = "Kết quả đề xuất" if top_prediction["probability"] < LOW_CONFIDENCE_THRESHOLD else "Kết quả"
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">{result_title}</div>
            <div class="result-value">{to_vi_label(top_prediction["class_name"])}</div>
            <div class="confidence-note">Độ tin cậy: {top_prediction["probability"] * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(top_prediction["probability"])

    if top_prediction["probability"] < LOW_CONFIDENCE_THRESHOLD:
        st.warning("Độ tin cậy thấp. Ảnh có thể khác phân phối dữ liệu huấn luyện; nên xem đủ 6 xác suất bên dưới.")

    if postprocessed:
        st.info("Ảnh có dấu hiệu là chai hoặc vật liệu nhựa; hệ thống đã dùng thêm kiểm tra phụ để hỗ trợ ảnh ngoài tập huấn luyện.")

    chart_df = pd.DataFrame(predictions)
    chart_df["nhóm rác"] = chart_df["class_name"].map(to_vi_label)
    chart_df["xác suất (%)"] = chart_df["probability"] * 100
    chart_df = chart_df.sort_values("xác suất (%)", ascending=False)

    st.subheader("Xác suất của 6 nhóm rác")
    st.bar_chart(chart_df.set_index("nhóm rác")["xác suất (%)"])
    st.dataframe(
        chart_df[["nhóm rác", "xác suất (%)"]].style.format({"xác suất (%)": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Hãy tải lên một ảnh rác để bắt đầu phân loại.")
