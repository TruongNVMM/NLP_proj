# Vietnamese Sentiment Analysis (CombViSA + Expand VSW)

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-4.30.0-yellow)

Dự án phân tích cảm xúc tiếng Việt kết hợp sức mạnh của mô hình ngôn ngữ PhoBERT và các đặc trưng từ điển từ SentiWordNet được mở rộng (CombViSA). Dự án này được cài đặt dựa trên một bài báo nghiên cứu về mở rộng Vietnamese Sentiwordnet. 

## Ý tưởng của việc mở rộng Vietnamese Sentiwordnet
1. **Seed extraction**: Trích xuất tập từ thuần thuần tích cực P và thuần tiêu cực N từ SentiWordNet dựa trên ngưỡng T. 
2. **Lexical expansion**: Mở rộng hai tập P và N bằng quan hệ đồng nghĩa / trái nghĩa từ WordNet và các từ điển phụ trợ như VCL (Vietnamese dictionaries).
3. **Score assignment**: Với mỗi từ mới, tính điểm sentiment bằng cách so sánh embedding của nó với tập P và N (xa gần đến tập P / N để gán pos/neg score).
$$PosScore = \frac{d_{pos}}{d_{pos} + d_{neg}}$$
$$NegScore = 1 - PosScore$$

## Tính năng chính

- **Mô hình lai (Hybrid approach)**: Kết hợp Deep Learning (PhoBERT) và Lexicon-based features (Vietnamese SentiWordNet).
- **Mở rộng từ điển**: Tự động mở rộng SentiWordNet tiếng Việt thông qua các quan hệ ngữ nghĩa (đồng nghĩa/phản nghĩa).
- **Hỗ trợ tập dữ liệu VSMEC**: Tích hợp sẵn bộ đọc dữ liệu cho Dataset phân tích cảm xúc tiếng Việt (Vietnamese Social Media Emotion Corpus).
- **Giao diện dự đoán**: Hỗ trợ dự đoán cảm xúc cho từng câu hoặc theo lô (batch).

## Cấu trúc dự án

- `train.py`: Tệp tin chính để huấn luyện mô hình.
- `commons.py`: Định nghĩa Dataset, Trainer và các hàm tiền xử lý văn bản.
- `config.py`: Cấu hình các tham số huấn luyện (batch size, learning rate, epochs, v.v.).
- `data/`: Thư mục chứa tập dữ liệu VSMEC (các file `.xlsx`).
- `modules/`:
  - `expandsentiwordnet.py`: Logic mở rộng từ điển cảm xúc.
  - `model.py`: Kiến trúc mô hình CombViSA.
- `requirements.txt`: Danh sách các thư viện cần thiết.

## Citation
```bibtex
@article{
  author    = {Hong-Viet Tran, Van-Tan Bui, Lam-Quan Tran},
  title     = {Expanding Vietnamese SentiWordNet to Improve Performance of Vietnamese Sentiment Analysis Models},
  year      = {2025},
  url       = {https://arxiv.org/abs/2501.08758},
  timestamp = {15/01/2025}
}
```
---
*Dự án được phát triển cho mục đích nghiên cứu NLP tiếng Việt của sinh viên.*
