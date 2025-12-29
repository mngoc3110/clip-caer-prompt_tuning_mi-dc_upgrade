# Tổng hợp các thay đổi nâng cấp dự án CLIP-CAER (Mục tiêu: 70% UAR)

Tài liệu này tóm tắt các thay đổi đã thực hiện đối với mã nguồn gốc để cải thiện độ chính xác (UAR) trên tập Validation, hướng tới việc huấn luyện trên nền tảng Kaggle.

## 1. Nâng cấp Kiến trúc Mô hình (Model Architecture)
**File:** `models/Generate_Model.py`

*   **Thay đổi:** Thay thế phương pháp nối đặc trưng (concatenation) đơn giản bằng cơ chế **Cross-Attention Fusion**.
*   **Chi tiết:**
    *   Tạo class `CrossAttention`.
    *   Sử dụng đặc trưng khuôn mặt (`face_feat`) làm **Query**.
    *   Sử dụng đặc trưng cơ thể/ngữ cảnh (`body_feat`) làm **Key** và **Value**.
*   **Tác dụng:** Giúp mô hình học được mối quan hệ sâu sắc hơn giữa biểu cảm khuôn mặt và ngữ cảnh cơ thể. Thay vì chỉ ghép 2 vector lại, mô hình giờ đây biết "chú ý" vào phần nào của ngữ cảnh để giải thích cho biểu cảm khuôn mặt, từ đó nhận diện chính xác hơn các ca khó (như mất tập trung nhưng mặt vẫn nhìn thẳng).

## 2. Tăng cường Dữ liệu (Data Augmentation)
**File:** `dataloader/video_transform.py` & `dataloader/video_dataloader.py`

*   **Thay đổi:** Thêm kỹ thuật biến đổi màu sắc `ColorJitter`.
*   **Chi tiết:**
    *   Thêm class `GroupTransform` và `ColorJitter` vào `video_transform.py`.
    *   Áp dụng `ColorJitter(brightness=0.1, contrast=0.1, ...)` vào quy trình `train_transforms` trong `video_dataloader.py`.
*   **Tác dụng:** Giúp mô hình không bị phụ thuộc vào điều kiện ánh sáng cụ thể của video gốc. Mô hình sẽ học đặc trưng biểu cảm thực sự thay vì học vẹt màu sắc của môi trường, làm tăng khả năng tổng quát hóa (Generalization) trên tập Validation.

## 3. Tinh chỉnh Siêu tham số & Cấu hình Kaggle
**File:** `kaggle_train.sh` (File mới)

*   **Thay đổi:** Tạo script chuyên biệt cho môi trường Kaggle GPU (P100/T4).
*   **Chi tiết cấu hình:**
    *   `--clip-path ViT-B/16`: Chuyển từ ViT-B/32 sang ViT-B/16. Patch size nhỏ hơn (16x16) giúp mô hình "nhìn" rõ chi tiết mắt, miệng hơn.
    *   `--epochs 50`: Tăng từ 20 lên 50 để đảm bảo Prompt Tuning hội tụ hoàn toàn.
    *   `--batch-size 16`: Tăng batch size để tận dụng VRAM của Kaggle P100, giúp train ổn định hơn (Batch Norm/Layer Norm hoạt động tốt hơn).
    *   `--label-smoothing 0.2`: Tăng từ 0.1 lên 0.2 để giảm hiện tượng Overfitting, giúp mô hình bớt tự tin thái quá vào các nhãn nhiễu.
    *   `--root-dir /kaggle/input/raer-video-emotion-dataset/RAER`: Cập nhật đường dẫn đúng chuẩn Kaggle Dataset.

## Tổng kết
Sự kết hợp của **Mô hình mạnh hơn (ViT-B/16)** + **Cơ chế Fusion thông minh (Cross-Attention)** + **Dữ liệu đa dạng hơn (ColorJitter)** + **Thời gian train lâu hơn (50 Epochs)** chính là công thức để đẩy UAR từ mức trung bình lên mức cao (~70%).
