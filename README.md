# Đồ án Deep Learning: Phân loại rác thải sinh hoạt hỗ trợ tái chế

Project này triển khai mô hình CNN bằng Python, NumPy, Pandas và PyTorch để phân loại ảnh rác thải sinh hoạt. Sau khi huấn luyện, mô hình được tích hợp vào ứng dụng web Streamlit để demo dự đoán ảnh thực tế.

File `Checklist đáp ứng yêu cầu đồ án.md` liệt kê rõ project đáp ứng từng yêu cầu của đề bài.

## 1. Cấu trúc thư mục

```text
waste_classifier/
  app.py
  requirements.txt
  data/
    train/
    val/
    test/
  outputs/
  src/
    data_utils.py
    metrics.py
    model.py
    plots.py
    predict.py
    train.py
```

## 2. Chuẩn bị dữ liệu

Dữ liệu cần đặt theo chuẩn `ImageFolder` của PyTorch:

```text
data/
  train/
    cardboard/
    glass/
    metal/
    paper/
    plastic/
    trash/
  val/
    cardboard/
    glass/
    metal/
    paper/
    plastic/
    trash/
  test/
    cardboard/
    glass/
    metal/
    paper/
    plastic/
    trash/
```

Bạn có thể dùng bộ dữ liệu TrashNet hoặc một bộ dữ liệu phân loại rác tương tự. Nếu dataset tải về chỉ có một thư mục ảnh theo lớp, hãy chia thủ công hoặc dùng script riêng để tách thành `train/val/test`.

Nếu dữ liệu ban đầu có dạng mỗi lớp là một thư mục, ví dụ `raw_dataset/plastic`, `raw_dataset/paper`, hãy chạy:

```bash
python src/split_dataset.py --input_dir raw_dataset --output_dir data
```

Tạo thống kê EDA cho báo cáo:

```bash
python src/eda.py --data_dir data
```

## 3. Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu máy có GPU NVIDIA và CUDA, nên cài PyTorch theo hướng dẫn chính thức để khớp phiên bản CUDA.

## 4. Huấn luyện mô hình

```bash
python src/train.py --data_dir data --epochs 20 --batch_size 32 --image_size 224
```

Nếu muốn kết quả tốt hơn cho demo, dùng transfer learning với MobileNetV2:

```bash
python src/train.py --data_dir data --epochs 10 --batch_size 32 --image_size 160 --architecture mobilenet_v2 --pretrained --class_weights --lr 0.001
```

Sau khi chạy xong, thư mục `outputs/` sẽ có:

- `best_model.pth`: model tốt nhất theo validation accuracy.
- `class_names.json`: danh sách nhãn.
- `training_history.csv`: lịch sử loss/accuracy từng epoch.
- `training_history.png`: biểu đồ train/validation.
- `test_classification_report.csv`: precision, recall, F1-score.
- `test_confusion_matrix.png`: ma trận nhầm lẫn.
- `metrics_summary.json`: tóm tắt chỉ số chính.

## 5. Dự đoán một ảnh bằng command line

```bash
python src/predict.py --model_path outputs/best_model.pth --image_path path/to/image.jpg
```

## 6. Chạy ứng dụng demo

```bash
streamlit run app.py
```

Ứng dụng cho phép tải ảnh lên, hiển thị nhãn dự đoán và xác suất top 3 lớp.

## 7. Nội dung có thể giải thích trong báo cáo

- Bài toán: phân loại ảnh rác sinh hoạt để hỗ trợ tái chế.
- Tiền xử lý: resize ảnh về `224x224`, chuyển tensor, chuẩn hóa theo mean/std.
- Tăng cường dữ liệu: lật ngang, xoay nhẹ, thay đổi sáng/tương phản.
- Kiến trúc: CNN tự xây gồm nhiều khối `Conv2D -> BatchNorm -> ReLU -> MaxPool`, hoặc MobileNetV2 transfer learning để demo chính xác hơn.
- Loss function: `CrossEntropyLoss`, phù hợp bài toán phân loại nhiều lớp.
- Optimizer: `Adam`, learning rate mặc định `0.001`.
- Đánh giá: Accuracy, Macro F1, Weighted F1, Classification Report, Confusion Matrix.
- Ứng dụng: web Streamlit dùng model đã huấn luyện để phân loại ảnh người dùng tải lên.

## 8. Lưu ý khi bảo vệ

Code này cố tình dùng CNN tự xây thay vì dùng model có sẵn để dễ trình bày bản chất. Nếu kết quả chưa cao, có thể cải thiện bằng cách tăng dữ liệu, cân bằng số ảnh mỗi lớp, tăng epoch hoặc thử transfer learning với ResNet/MobileNet.
