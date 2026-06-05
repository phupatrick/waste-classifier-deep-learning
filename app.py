from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from src.predict import load_model, predict_image


st.set_page_config(page_title="Phân loại rác tái chế", page_icon="♻️", layout="wide")

MODEL_PATH = Path("outputs/best_model.pth")
LOW_CONFIDENCE_THRESHOLD = 0.5

CLASS_LABELS_VI = {
    "cardboard": "Bìa carton",
    "glass": "Thủy tinh",
    "metal": "Kim loại",
    "paper": "Giấy",
    "plastic": "Nhựa",
    "trash": "Rác khác",
}

CLASS_DESCRIPTIONS = {
    "cardboard": "Hộp giấy, bìa cứng",
    "glass": "Chai, lọ thủy tinh",
    "metal": "Lon, nắp, kim loại",
    "paper": "Giấy, báo, tài liệu",
    "plastic": "Chai, túi, hộp nhựa",
    "trash": "Rác khó tái chế",
}

CLASS_COLORS = {
    "cardboard": "#A16207",
    "glass": "#0284C7",
    "metal": "#475569",
    "paper": "#2563EB",
    "plastic": "#15803D",
    "trash": "#7C3AED",
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


def class_color(class_name: str) -> str:
    return CLASS_COLORS.get(class_name, "#2563EB")


st.markdown(
    """
    <style>
    :root {
        --bg: #f3f6f8;
        --surface: #ffffff;
        --ink: #0f172a;
        --muted: #64748b;
        --line: #dbe3ea;
        --soft: #eef4f7;
        --navy: #102033;
        --green: #15803d;
        --blue: #2563eb;
        --amber: #b7791f;
    }
    .stApp {
        background:
            linear-gradient(180deg, #edf3f6 0%, #f8fafc 42%, #f3f6f8 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1180px;
        padding: 1.25rem 1.4rem 2rem 1.4rem;
    }
    h1, h2, h3, p {
        letter-spacing: 0 !important;
    }
    h1 {
        font-size: 2rem !important;
        line-height: 1.15 !important;
        margin: 0 !important;
    }
    h2, h3 {
        font-size: 1.12rem !important;
        line-height: 1.3 !important;
    }
    [data-testid="stSidebar"] {
        background: #102033;
    }
    [data-testid="stSidebar"] * {
        color: #e5edf5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }
    [data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px dashed #9fb1c3;
        border-radius: 8px;
        padding: 12px 14px 4px 14px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid var(--line);
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.10);
    }
    .app-header {
        background: #102033;
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 16px 38px rgba(15, 23, 42, 0.16);
        position: relative;
        overflow: hidden;
    }
    .app-header:before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 0;
        height: 4px;
        background: linear-gradient(90deg, #15803d, #2563eb, #b7791f, #7c3aed);
    }
    .header-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 18px;
        align-items: center;
    }
    .eyebrow {
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .header-title {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 850;
        line-height: 1.15;
    }
    .header-subtitle {
        color: #cbd5e1;
        font-size: 0.98rem;
        margin-top: 6px;
        max-width: 680px;
    }
    .status-pill {
        border: 1px solid rgba(187, 247, 208, 0.30);
        background: rgba(22, 163, 74, 0.16);
        color: #dcfce7;
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 0.86rem;
        font-weight: 800;
        white-space: nowrap;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 14px;
    }
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 13px 14px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }
    .stat-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .stat-value {
        color: var(--ink);
        font-size: 1.22rem;
        font-weight: 850;
        margin-top: 2px;
    }
    .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    }
    .section-label {
        color: #475569;
        font-size: 0.8rem;
        font-weight: 850;
        text-transform: uppercase;
        margin-bottom: 9px;
    }
    .result-card {
        border-radius: 8px;
        padding: 18px;
        background:
            linear-gradient(135deg, rgba(21, 128, 61, 0.12), rgba(37, 99, 235, 0.10)),
            #ffffff;
        border: 1px solid #b7d7c6;
        margin-bottom: 12px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
    }
    .result-kicker {
        color: #475569;
        font-size: 0.84rem;
        font-weight: 850;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .result-main {
        color: #0f172a;
        font-size: 2.15rem;
        font-weight: 900;
        line-height: 1.12;
        margin-bottom: 10px;
    }
    .confidence {
        color: #334155;
        font-size: 0.98rem;
    }
    .warn-box, .hint-box {
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 0.92rem;
        margin: 10px 0;
        line-height: 1.35;
    }
    .warn-box {
        border: 1px solid #facc15;
        background: #fefce8;
        color: #854d0e;
    }
    .hint-box {
        border: 1px solid #93c5fd;
        background: #eff6ff;
        color: #1e3a8a;
    }
    .class-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 14px 0;
    }
    .class-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 11px 10px;
        min-height: 84px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }
    .class-mark {
        width: 28px;
        height: 5px;
        border-radius: 999px;
        margin-bottom: 9px;
    }
    .class-name {
        color: #0f172a;
        font-size: 0.95rem;
        font-weight: 850;
        margin-bottom: 4px;
    }
    .class-desc {
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.25;
    }
    .prob-row {
        display: grid;
        grid-template-columns: minmax(100px, 140px) 1fr 68px;
        gap: 12px;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #eef2f7;
    }
    .prob-row:last-child {
        border-bottom: none;
    }
    .prob-label {
        color: #1f2937;
        font-weight: 800;
        font-size: 0.92rem;
    }
    .prob-track {
        height: 12px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
    }
    .prob-value {
        color: #334155;
        font-variant-numeric: tabular-nums;
        text-align: right;
        font-size: 0.9rem;
        font-weight: 800;
    }
    .upload-note {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: -2px;
        margin-bottom: 8px;
    }
    @media (max-width: 960px) {
        .header-grid, .stats-grid, .class-grid {
            grid-template-columns: 1fr 1fr;
        }
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        .header-grid, .stats-grid, .class-grid {
            grid-template-columns: 1fr;
        }
        .header-title {
            font-size: 1.55rem;
        }
        .result-main {
            font-size: 1.6rem;
        }
        .prob-row {
            grid-template-columns: 94px 1fr 58px;
            gap: 8px;
        }
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

    adjusted = [dict(item) for item in predictions]
    class_names = {item["class_name"] for item in adjusted}
    if "plastic" not in class_names:
        adjusted.append({"class_name": "plastic", "probability": 0.0})

    for item in adjusted:
        if item["class_name"] == "plastic":
            item["probability"] = max(item["probability"], 0.56)

    total = sum(item["probability"] for item in adjusted)
    for item in adjusted:
        item["probability"] = item["probability"] / total

    return sorted(adjusted, key=lambda item: item["probability"], reverse=True), True


def render_class_grid():
    cards = []
    for class_name in CLASS_LABELS_VI:
        cards.append(
            f"""
            <div class="class-card">
                <div class="class-mark" style="background: {class_color(class_name)};"></div>
                <div class="class-name">{to_vi_label(class_name)}</div>
                <div class="class-desc">{CLASS_DESCRIPTIONS[class_name]}</div>
            </div>
            """
        )
    return "\n".join(cards)


def render_probability_rows(predictions):
    rows = []
    for item in predictions:
        percent = item["probability"] * 100
        label = to_vi_label(item["class_name"])
        color = class_color(item["class_name"])
        rows.append(
            f"""
            <div class="prob-row">
                <div class="prob-label">{label}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {percent:.2f}%; background: {color};"></div>
                </div>
                <div class="prob-value">{percent:.2f}%</div>
            </div>
            """
        )
    return "\n".join(rows)


with st.sidebar:
    st.markdown("### Thông tin mô hình")
    st.markdown("**Kiến trúc:** MobileNetV2")
    st.markdown("**Framework:** PyTorch")
    st.markdown("**Dataset:** TrashNet")
    st.markdown("**Số lớp:** 6")
    st.divider()
    st.markdown("### Chỉ số")
    st.markdown("**Accuracy:** 84.11%")
    st.markdown("**Macro F1:** 80.86%")
    st.markdown("**Weighted F1:** 84.00%")

st.markdown(
    """
    <div class="app-header">
        <div class="header-grid">
            <div>
                <div class="eyebrow">Ứng dụng phân loại rác tái chế</div>
                <div class="header-title">Phân loại rác thải sinh hoạt</div>
                <div class="header-subtitle">Nhận diện 6 nhóm rác từ ảnh đầu vào bằng mô hình MobileNetV2 đã huấn luyện trên TrashNet.</div>
            </div>
            <div class="status-pill">Model sẵn sàng</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Số nhóm</div>
            <div class="stat-value">6 lớp</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Accuracy</div>
            <div class="stat-value">84.11%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Weighted F1</div>
            <div class="stat-value">84.00%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Đầu vào</div>
            <div class="stat-value">Ảnh RGB</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="class-grid">', unsafe_allow_html=True)
st.markdown(render_class_grid(), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.warning("Chưa tìm thấy mô hình đã huấn luyện. Hãy chạy lệnh train trước để tạo file outputs/best_model.pth.")
    st.code(
        "python src/train.py --data_dir data --epochs 10 --batch_size 32 --image_size 160 "
        "--architecture mobilenet_v2 --pretrained --class_weights --lr 0.001",
        language="bash",
    )
    st.stop()

st.markdown('<div class="section-label">Tải ảnh kiểm thử</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-note">Hỗ trợ JPG, JPEG, PNG và WEBP.</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Tải ảnh rác cần phân loại",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    temp_dir = Path("outputs/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / uploaded_file.name
    image.save(temp_path)

    model, class_names, image_size, device = get_waste_model(str(MODEL_PATH))
    predictions = predict_image(model, str(temp_path), class_names, image_size, device, top_k=6)
    predictions, postprocessed = apply_low_confidence_postprocess(predictions, image)
    top_prediction = predictions[0]
    confidence = top_prediction["probability"] * 100
    result_title = "Kết quả đề xuất" if top_prediction["probability"] < LOW_CONFIDENCE_THRESHOLD else "Kết quả"

    left_col, right_col = st.columns([1.05, 0.95], gap="large")
    with left_col:
        st.markdown('<div class="section-label">Ảnh đầu vào</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with right_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-kicker">{result_title}</div>
                <div class="result-main">{to_vi_label(top_prediction["class_name"])}</div>
                <div class="confidence">Độ tin cậy: <strong>{confidence:.2f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(top_prediction["probability"])

        if top_prediction["probability"] < LOW_CONFIDENCE_THRESHOLD:
            st.markdown(
                '<div class="warn-box">Độ tin cậy thấp. Nên xem đủ 6 xác suất bên dưới trước khi kết luận.</div>',
                unsafe_allow_html=True,
            )

        if postprocessed:
            st.markdown(
                '<div class="hint-box">Ảnh có dấu hiệu chai hoặc vật liệu nhựa nên hệ thống dùng thêm kiểm tra phụ.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Xác suất của 6 nhóm rác</div>', unsafe_allow_html=True)
        st.markdown(render_probability_rows(predictions), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    chart_df = pd.DataFrame(predictions)
    chart_df["Nhóm rác"] = chart_df["class_name"].map(to_vi_label)
    chart_df["Xác suất (%)"] = chart_df["probability"] * 100
    chart_df = chart_df.sort_values("Xác suất (%)", ascending=False)

    with st.expander("Bảng số liệu chi tiết"):
        st.dataframe(
            chart_df[["Nhóm rác", "Xác suất (%)"]].style.format({"Xác suất (%)": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Tải ảnh lên để bắt đầu phân loại.")
