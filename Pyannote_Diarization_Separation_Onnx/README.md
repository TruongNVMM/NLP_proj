# Pyannote Diarization & Separation hỗ trợ model onnx

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-EE4C2C?logo=pytorch&logoColor=white)
![pyannote](https://img.shields.io/badge/pyannote.audio-3.0+-4CAF50?logo=pyannote&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-1.16+-4CAF50?logo=onnx&logoColor=white)

> Một hệ thống pipeline mạnh mẽ kết hợp **Phân tách người nói (Speaker Diarization)** và **Tách lọc giọng nói (Speech Separation)** sử dụng thư viện `pyannote.audio` cùng với các mô hình ONNX đã được tối ưu hóa.

Dự án này cung cấp một môi trường hoàn chỉnh và kịch bản tự động hóa để thực hiện hai nhiệm vụ: **speaker diarization** (xác định ai đang nói và vào lúc nào) và **speech separation** (tách riêng từng bản ghi giọng nói của các diễn giả từ một file âm thanh hỗn hợp). Chúng tôi tận dụng sức mạnh của các mô hình tiên tiến từ `pyannote.audio` và chuyển đổi chúng sang định dạng `ONNX` để tăng tốc độ xử lý trên mọi nền tảng—ngay cả đối với các thiết bị không có GPU hỗ trợ CUDA.

---
## Nguồn tham khảo
**Dự án này có tham khảo và sử dụng toàn bộ dự án **pyannote audio** và nhóm đã phát triển tái cấu trúc và sửa lại code để pù hợp với dự án của lab nghiên cứu: tích hợp cả pyannote audio diarization, speech separation và onnx để có thể chạy trên đa nền tảng và các thiết bị nhúng: Jetson Nano/Orin hay Raspberry Pi.**

     https://github.com/pyannote/pyannote-audio

## Các tính năng chính

* **Diarization độ chính xác cao**: Xây dựng dựa trên khả năng phân đoạn (segmentation) và nhúng (embedding) xuất sắc của `pyannote.audio`.
* **Khả năng tách đoạn hội thoại**: Không chỉ dừng lại ở việc tạo nhãn thời gian (timestamps) mà còn xuất khẩu vật lý (extract) từng giọng nói riêng biệt từ file âm thanh gồm nhiều người nói.
* **Tăng tốc với ONNX**: Sử dụng `onnxruntime` để cải thiện đáng kể tốc độ suy luận (inference speed) với các tệp định dạng `.onnx` và tệp lượng tử hóa `.int8.onnx`.
* **Kịch bản tự động (Scripts)**: Cung cấp sẵn các luồng làm việc (workflow) với Python giúp dễ dàng chuyển đổi các mô hình PyTorch sang định dạng ONNX.
* **Mã nguồn Module hóa (Modular Codebase)**: Được thiết kế với kiến trúc dễ dàng tích hợp vào các công cụ Downstream. Hệ thống thư mục được chia tách rõ ràng: pipeline chính, công cụ tiện ích (utils) và script kiểm thử.

---

## Cấu trúc thư mục dự án

Kho lưu trữ này được cấu trúc theo tiêu chuẩn mã nguồn mở Python chuyên nghiệp:

```text
pyannote_diarization/
├── src/                        # ➔ Mã nguồn cốt lõi và thư viện logic chính 
│   ├── diarization_pipeline/   # Các class thiết lập pipeline kết hợp (Combined Pipeline)
│   └── sherpa_onnx_utils/      # Các tiện ích liên quan đến VAD, segmentation của sherpa-onnx
├── models/                     # ➔ Nơi lưu trữ trọng số mô hình (VD: ONNX, PyTorch checkpoints)
├── examples/                   # ➔ Thư mục thử nghiệm và kịch bản ví dụ (cách sử dụng pipeline)
├── scripts/                    # ➔ Script phục vụ DevOps và Testing (chuyển đổi ONNX, test tự động)
├── docs/                       # ➔ Chi tiết tài liệu nâng cao, hướng dẫn và thông tin bản quyền (License)
├── analysis/                   # ➔ Phân tích dữ liệu, sổ tay Jupyter notebooks và file âm thanh chạy thử
├── third_party/                # ➔ Chứa các thư viện bên thứ ba tự dựng (VD: mã clone từ pyannote-audio)
└── requirements.txt            # ➔ Các package dependency bắt buộc của Python
```

---

## Cài đặt & Thiết lập

Chúng tôi khuyến nghị bạn nên sử dụng môi trường ảo hóa (virtual environment) để tránh xung đột thư viện.

**1. Tạo & Kích hoạt Môi trường ảo:**
```bash
# Trên Windows
python -m venv venv
.\venv\Scripts\activate

# Trên Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

**2. Cài đặt Dependencies:**
```bash
pip install -r requirements.txt
```
*(Nếu máy của bạn có GPU hỗ trợ CUDA, hãy đảm bảo cài đặt chuẩn xác phiên bản `onnxruntime-gpu` và `torch` tương ứng theo cấu hình hệ thống của bạn).*

---

## Hướng dẫn Bắt đầu nhanh & Kết xuất

### 1. Chuẩn bị Mô hình & Kiểm tra Thiết lập ban đầu
Để lấy tài nguyên sẵn sàng (download requirements, xác nhận môi trường, chuyển đổi PyTorch models sang ONNX), bạn chỉ cần chạy tập lệnh (script) tự động sau:
```bash
python scripts/setup_and_test.py
```

### 2. Chạy Pipeline Ví dụ (Example Pipeline)
Hãy mở và chạy tệp `examples/example_usage.py` để xem minh họa hệ thống tách lọc làm việc. Nó sẽ tự sinh ra đoạn âm thanh giả lập chồng chéo người nói và bóc tách chúng ra.

```bash
python examples/example_usage.py
```
Tiến trình kịch bản này sẽ làm là:
1. Đọc và tải file mô hình `.int8.onnx` từ thư mục `models/`.
2. Khởi tạo Engine suy luận từ Pyannote.
3. Chạy quá trình quét trên file âm thanh `wav`.
4. Xuất ra các âm thanh rời rạc (separation tracks) và xuất JSON chứa timestamps.

---

## Phân giải Mô hình Sang ONNX thủ công (Manual Export)

Nếu bạn mang đến tệp `.bin` chạy trên PyTorch của riêng bạn và muốn tối ưu hóa chúng sang định dạng ONNX gọn nhẹ, bạn có thể thực hiện thông qua module `scripts/`:

```bash
# Đặt file 'pytorch_model.bin' của bạn vào đúng dự án rồi gọi:
python scripts/export_pyannote_separation.py
```
Hoạt động này sẽ kết xuất tệp mới `model.onnx` và `model.int8.onnx` thích hợp nhất cho quá trình tiêu hao bộ nhớ thấp.

---

