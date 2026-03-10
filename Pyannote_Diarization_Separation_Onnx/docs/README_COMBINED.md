# Combined Pipeline: Speaker Diarization + Speech Separation

Pipeline kết hợp **Speaker Diarization** và **Speech Separation** để tạo ra:
- ✅ Timestamps của từng speaker (diarization)
- ✅ Audio tách nguồn cho từng speaker (separated sources)
- ✅ Hỗ trợ ONNX model cho segmentation (model.int8.onnx)

## Yêu cầu

```bash
pip install pyannote.audio soundfile torch onnxruntime
```

Nếu có GPU:
```bash
pip install pyannote.audio soundfile torch onnxruntime-gpu
```

## Cách sử dụng

### 1. Sử dụng Command Line

```bash
python combined_pipeline.py audio.wav --output-dir output/
```

**Với các tùy chọn:**

```bash
python combined_pipeline.py audio.wav \
    --segmentation-model model.int8.onnx \
    --num-speakers 3 \
    --output-dir my_output/ \
    --hf-token YOUR_HUGGINGFACE_TOKEN
```

**Tham số:**
- `audio_file`: Đường dẫn file audio input (bắt buộc)
- `--segmentation-model`: Đường dẫn ONNX model (mặc định: `model.int8.onnx`)
- `--num-speakers`: Số speakers nếu biết trước
- `--min-speakers`: Số speakers tối thiểu
- `--max-speakers`: Số speakers tối đa
- `--output-dir`: Thư mục output (mặc định: `output/`)
- `--hf-token`: HuggingFace token để download models

### 2. Sử dụng trong Python Code

```python
from combined_pipeline import CombinedPipeline

# Khởi tạo pipeline
pipeline = CombinedPipeline(
    segmentation_model="model.int8.onnx",
    use_auth_token="YOUR_HF_TOKEN"  # optional
)

# Xử lý audio
result = pipeline(
    "audio.wav",
    num_speakers=3  # optional
)

# Truy cập kết quả
print("Timestamps:", result.timestamps)
print("Speakers:", result.separated_sources.keys())

# Lưu kết quả
result.save_separated_audio("output/")
result.save_timestamps("output/timestamps.json")
result.save_timestamps("output/timestamps.rttm")
```

## Cấu trúc Output

### CombinedOutput Object

```python
result = pipeline("audio.wav")

# Diarization annotation
result.diarization  # pyannote.core.Annotation object

# Timestamps (list of dicts)
result.timestamps = [
    {'start': 0.5, 'end': 2.3, 'duration': 1.8, 'speaker': 'SPEAKER_00'},
    {'start': 2.5, 'end': 4.1, 'duration': 1.6, 'speaker': 'SPEAKER_01'},
    ...
]

# Separated audio sources
result.separated_sources = {
    'SPEAKER_00': numpy_array_audio,
    'SPEAKER_01': numpy_array_audio,
    ...
}

# Sample rate
result.sample_rate  # 16000
```

### Output Files

Khi chạy `result.save_separated_audio("output/")`:
```
output/
├── SPEAKER_00.wav
├── SPEAKER_01.wav
├── SPEAKER_02.wav
└── ...
```

Khi chạy `result.save_timestamps("output/timestamps.json")`:
```json
[
  {
    "start": 0.5,
    "end": 2.3,
    "duration": 1.8,
    "speaker": "SPEAKER_00"
  },
  ...
]
```

Khi chạy `result.save_timestamps("output/timestamps.rttm")`:
```
SPEAKER file 1 0.500 1.800 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER file 1 2.500 1.600 <NA> <NA> SPEAKER_01 <NA> <NA>
...
```

## Kiến trúc Pipeline

```
┌─────────────────────────────────────────────────────┐
│                  Combined Pipeline                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │  Step 1: Speaker Diarization               │    │
│  │  - Segmentation (ONNX model.int8.onnx)     │    │
│  │  - Embedding                                │    │
│  │  - Clustering                               │    │
│  │  → Output: Timestamps + Speaker Labels     │    │
│  └────────────────────────────────────────────┘    │
│                       ↓                             │
│  ┌────────────────────────────────────────────┐    │
│  │  Step 2: Speech Separation                 │    │
│  │  - Separation Model                        │    │
│  │  → Output: Separated Audio Sources         │    │
│  └────────────────────────────────────────────┘    │
│                       ↓                             │
│  ┌────────────────────────────────────────────┐    │
│  │  Step 3: Combine Results                   │    │
│  │  - Match speakers với separated sources    │    │
│  │  - Extract audio cho mỗi speaker           │    │
│  │  → Output: {timestamps, separated_audio}   │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Tùy chỉnh

### Sử dụng Custom Models

```python
pipeline = CombinedPipeline(
    segmentation_model="path/to/your/model.int8.onnx",
    embedding_model="pyannote/embedding",
    separation_model="pyannote/separation-ami-1.0"
)
```

### Xử lý Batch Files

```python
from pathlib import Path

audio_files = Path("audio_folder").glob("*.wav")

for audio_file in audio_files:
    result = pipeline(str(audio_file))
    
    # Save với tên file gốc
    output_dir = Path("output") / audio_file.stem
    result.save_separated_audio(output_dir)
    result.save_timestamps(output_dir / "timestamps.json")
```

## Notes

1. **ONNX Model**: File `model.int8.onnx` cần được đặt ở cùng thư mục hoặc chỉ định đường dẫn đầy đủ
2. **HuggingFace Token**: Một số models cần token để download. Lấy token tại [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **GPU Support**: Pipeline tự động sử dụng GPU nếu có sẵn
4. **Memory**: Với audio dài, pipeline có thể tốn nhiều RAM. Cần theo dõi memory usage

## Troubleshooting

### Lỗi "Segmentation model not found"
```
❌ Error: Model file not found at model.int8.onnx
✅ Solution: Đảm bảo file model.int8.onnx tồn tại hoặc chỉ định đường dẫn đúng
```

### Lỗi import pyannote
```
❌ Error: No module named 'pyannote'
✅ Solution: pip install pyannote.audio
```

### Out of Memory
```
❌ Error: CUDA out of memory
✅ Solution: 
   - Giảm batch size
   - Sử dụng CPU thay vì GPU
   - Chia nhỏ audio file
```

## Tài liệu tham khảo

- [pyannote.audio documentation](https://github.com/pyannote/pyannote-audio)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Paper: pyannote.audio speaker diarization](https://arxiv.org/abs/2012.01255)
