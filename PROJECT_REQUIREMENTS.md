# Checklist đáp ứng yêu cầu đồ án

Đề tài: Phân loại rác thải sinh hoạt hỗ trợ tái chế.

## 1. Yêu cầu kỹ thuật

Project sử dụng Python và các thư viện bắt buộc:

- NumPy: dùng trong `src/train.py`, `src/metrics.py`, `src/plots.py`.
- Pandas: dùng trong `src/train.py`, `src/eda.py`, `src/metrics.py`, `app.py`.
- PyTorch: dùng trong `src/model.py`, `src/train.py`, `src/predict.py`.

Các thư viện hỗ trợ:

- Torchvision: đọc dataset ảnh và tiền xử lý ảnh.
- Scikit-learn: tính Accuracy, F1-score, Classification Report, Confusion Matrix.
- Matplotlib: vẽ biểu đồ huấn luyện và confusion matrix.
- Streamlit: xây dựng ứng dụng web demo.

## 2. Kiến trúc mô hình

Project triển khai mô hình CNN trong `src/model.py`.

Kiến trúc chính:

- Nhiều khối `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d`.
- `AdaptiveAvgPool2d` để gom đặc trưng.
- `Dropout` để giảm overfitting.
- `Linear` để phân loại 6 lớp rác.

Các lớp rác:

- cardboard
- glass
- metal
- paper
- plastic
- trash

## 3. Huấn luyện và đánh giá

File huấn luyện chính: `src/train.py`.

Quy trình:

- Đọc dữ liệu theo cấu trúc `data/train`, `data/val`, `data/test`.
- Resize ảnh, chuẩn hóa ảnh, tăng cường dữ liệu cho tập train.
- Huấn luyện bằng `CrossEntropyLoss`.
- Tối ưu bằng `Adam`.
- Lưu model tốt nhất theo validation accuracy.
- Đánh giá trên test set bằng Accuracy, Macro F1, Weighted F1, Classification Report và Confusion Matrix.

Kết quả train nhanh hiện tại:

- Best validation accuracy: 61.80%.
- Test accuracy: 58.07%.
- Macro F1: 51.04%.
- Weighted F1: 55.84%.

Các file kết quả nằm trong `outputs/`.

## 4. EDA dữ liệu

File EDA: `src/eda.py`.

EDA tạo:

- Danh sách ảnh trong dataset.
- Thống kê số lượng ảnh theo tập train/val/test và từng lớp.
- Biểu đồ phân bố lớp.

Kết quả nằm trong `outputs/eda/`.

## 5. Ứng dụng thực tế

Ứng dụng demo: `app.py`.

Ứng dụng cho phép:

- Upload ảnh rác từ máy người dùng.
- Hiển thị ảnh đầu vào.
- Dự đoán lớp rác bằng model đã train.
- Hiển thị độ tin cậy và top dự đoán.

Chạy app:

```bash
streamlit run app.py
```

## 6. Tính chính trực

Code trong project được tổ chức theo từng module rõ ràng để sinh viên có thể giải thích:

- `src/model.py`: định nghĩa kiến trúc CNN.
- `src/data_utils.py`: tiền xử lý ảnh và DataLoader.
- `src/train.py`: vòng lặp huấn luyện.
- `src/metrics.py`: tính chỉ số đánh giá.
- `src/predict.py`: dự đoán ảnh mới.
- `app.py`: tích hợp model vào web app.

Không dùng notebook hoặc source code Kaggle của người khác làm code chính. Dataset chỉ được dùng làm dữ liệu huấn luyện.
