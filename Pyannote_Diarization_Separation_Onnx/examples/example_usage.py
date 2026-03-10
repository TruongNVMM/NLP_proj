import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import soundfile as sf
from diarization_pipeline.combined_pipeline import CombinedPipeline, CombinedOutput

def create_sample_audio(output_path: str = "sample_audio.wav", duration: float = 10.0):
    """
    Tạo một file audio mẫu để test
    
    Parameters
    ----------
    output_path : str
        Đường dẫn output file
    duration : float
        Độ dài audio (giây)
    """
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Tạo 2 tín hiệu khác nhau (giả lập 2 speakers)
    freq1 = 440  # A4 note
    freq2 = 554  # C#5 note
    
    # Speaker 1: nửa đầu
    signal1 = np.sin(2 * np.pi * freq1 * t[:len(t)//2])
    # Speaker 2: nửa sau
    signal2 = np.sin(2 * np.pi * freq2 * t[len(t)//2:])
    
    # Combine
    audio = np.concatenate([signal1, signal2])
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.5
    
    sf.write(output_path, audio, sample_rate)
    print(f"Created sample audio: {output_path}")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sample_rate} Hz")
    
    return output_path


def run_example():
    """
    Chạy example với combined pipeline
    """
    print("="*70)
    print("COMBINED PIPELINE EXAMPLE")
    print("="*70)
    print()
    
    # 1. Tạo sample audio (hoặc sử dụng audio có sẵn)
    print("Step 1: Preparing audio...")
    audio_file = "sample_audio.wav"
    
    # Uncomment để tạo sample audio mới
    # create_sample_audio(audio_file, duration=10.0)
    
    # Kiểm tra xem file có tồn tại không
    if not Path(audio_file).exists():
        print(f"Audio file not found: {audio_file}")
        print("Creating sample audio...")
        create_sample_audio(audio_file, duration=10.0)
    else:
        print(f"Using existing audio: {audio_file}")
    
    print()
    
    # 2. Khởi tạo pipeline
    print("Step 2: Initializing pipeline...")
    try:
        pipeline = CombinedPipeline(
            segmentation_model=str(Path(__file__).resolve().parent.parent / "models" / "model.int8.onnx"),  # Sẽ fallback về pretrained nếu không tìm thấy
            # use_auth_token="YOUR_HF_TOKEN"  # Uncomment nếu cần
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Trying with default settings...")
        pipeline = CombinedPipeline()
    
    print()
    
    # 3. Chạy pipeline
    print("Step 3: Running pipeline...")
    try:
        result = pipeline(
            audio_file,
            min_speakers=1,
            max_speakers=5
        )
        
        # 4. Hiển thị kết quả
        print()
        print("="*70)
        print("RESULTS")
        print("="*70)
        
        print(f"\nNumber of speakers: {len(result.separated_sources)}")
        print(f"Total segments: {len(result.timestamps)}")
        print(f"Sample rate: {result.sample_rate} Hz")
        
        print("\nTimestamps:")
        for i, ts in enumerate(result.timestamps[:10], 1):
            print(f"  {i:2d}. {ts['start']:6.2f}s - {ts['end']:6.2f}s "
                  f"({ts['duration']:5.2f}s) → {ts['speaker']}")
        
        if len(result.timestamps) > 10:
            print(f"  ... and {len(result.timestamps) - 10} more segments")
        
        print("\n🔊 Separated Sources:")
        for speaker, audio in result.separated_sources.items():
            duration = len(audio) / result.sample_rate
            print(f"  {speaker}: {len(audio)} samples ({duration:.2f}s)")
        
        # 5. Lưu kết quả
        print("\nStep 4: Saving results...")
        output_dir = Path("example_output")
        output_dir.mkdir(exist_ok=True)
        
        # Lưu separated audio
        audio_files = result.save_separated_audio(output_dir)
        print(f"  Saved {len(audio_files)} audio files")
        
        # Lưu timestamps
        result.save_timestamps(output_dir / "timestamps.json")
        result.save_timestamps(output_dir / "timestamps.rttm")
        print(f"  Saved timestamps to JSON and RTTM")
        
        # Get dict representation
        summary = result.to_dict()
        print("\nSummary:")
        import json
        print(json.dumps(summary, indent=2))
        
        print("\n" + "="*70)
        print(f"SUCCESS! Results saved to: {output_dir.absolute()}")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def run_batch_example():
    """
    Example xử lý nhiều files
    """
    print("="*70)
    print("BATCH PROCESSING EXAMPLE")
    print("="*70)
    print()
    
    # Tạo sample audio files
    audio_dir = Path("sample_audios")
    audio_dir.mkdir(exist_ok=True)
    
    print("Creating sample audio files...")
    audio_files = []
    for i in range(3):
        audio_path = audio_dir / f"audio_{i+1}.wav"
        create_sample_audio(str(audio_path), duration=5.0 + i*2)
        audio_files.append(audio_path)
    
    print()
    
    # Initialize pipeline
    pipeline = CombinedPipeline()
    
    # Process each file
    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file.name}")
        try:
            result = pipeline(str(audio_file))
            
            # Save results
            output_dir = Path("batch_output") / audio_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result.save_separated_audio(output_dir)
            result.save_timestamps(output_dir / "timestamps.json")
            
            print(f"Saved to {output_dir}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*70)
    print("Batch processing completed!")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined Pipeline Examples")
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Example mode: single file or batch processing"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        success = run_example()
        exit(0 if success else 1)
    
    elif args.mode == "batch":
        run_batch_example()
