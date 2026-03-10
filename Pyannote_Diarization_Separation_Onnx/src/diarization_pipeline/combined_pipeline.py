"""
Combined Pipeline: Speaker Diarization + Speech Separation
Kết hợp speaker diarization với speech separation để tạo ra:
- Timestamps của từng speaker (diarization)
- Audio tách nguồn cho từng speaker (separation)
"""

import warnings
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import numpy as np
import torch
from dataclasses import dataclass

try:
    from pyannote.audio import Pipeline, Audio
    from pyannote.audio.core.io import AudioFile
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    from pyannote.audio.pipelines.speech_separation import SpeechSeparation
    from pyannote.core import Annotation, Segment
    import soundfile as sf
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install: pip install pyannote.audio soundfile torch")
    raise


@dataclass
class CombinedOutput:
    """Output structure cho combined pipeline"""
    # Diarization results
    diarization: Annotation  # Timeline với speaker labels
    timestamps: List[Dict]  # List of {start, end, speaker}
    
    # Separation results
    separated_sources: Dict[str, np.ndarray]  # {speaker_id: audio_array}
    sample_rate: int
    
    def save_separated_audio(self, output_dir: Union[str, Path]):
        """
        Lưu các audio tách nguồn vào thư mục
        
        Parameters
        ----------
        output_dir : str or Path
            Thư mục đích để lưu các file audio
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for speaker_id, audio_data in self.separated_sources.items():
            output_path = output_dir / f"{speaker_id}.wav"
            sf.write(output_path, audio_data, self.sample_rate)
            saved_files.append(str(output_path))
            print(f"Saved {speaker_id} audio to: {output_path}")
        
        return saved_files
    
    def save_timestamps(self, output_path: Union[str, Path]):
        """
        Lưu timestamps ra file JSON hoặc RTTM
        
        Parameters
        ----------
        output_path : str or Path
            Đường dẫn file output
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.json':
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.timestamps, f, indent=2, ensure_ascii=False)
            print(f"Saved timestamps to: {output_path}")
        
        elif output_path.suffix == '.rttm':
            # RTTM format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
            with open(output_path, 'w') as f:
                for segment in self.timestamps:
                    start = segment['start']
                    duration = segment['end'] - segment['start']
                    speaker = segment['speaker']
                    f.write(f"SPEAKER file 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")
            print(f"Saved RTTM to: {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}. Use .json or .rttm")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization"""
        return {
            'timestamps': self.timestamps,
            'speakers': list(self.separated_sources.keys()),
            'sample_rate': self.sample_rate,
            'num_speakers': len(self.separated_sources)
        }


class CombinedPipeline:
    """
    Combined Pipeline kết hợp Speaker Diarization và Speech Separation
    
    Parameters
    ----------
    segmentation_model : str or Path
        Đường dẫn đến model ONNX cho segmentation (ví dụ: model.int8.onnx)
    embedding_model : str, optional
        Model cho speaker embedding
    separation_model : str, optional
        Model cho speech separation
    use_auth_token : str, optional
        HuggingFace token nếu cần
    device : torch.device, optional
        Device để chạy model (cpu hoặc cuda)
    """
    
    def __init__(
        self,
        segmentation_model: Union[str, Path] = "model.int8.onnx",
        embedding_model: Optional[str] = None,
        separation_model: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        if str(segmentation_model) == "model.int8.onnx":
            segmentation_model = Path(__file__).resolve().parent.parent.parent / "models" / "model.int8.onnx"
        self.segmentation_model = Path(segmentation_model)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio = Audio(sample_rate=16000, mono=True)
        
        print(f"Initializing Combined Pipeline on {self.device}")
        print(f"Segmentation model: {self.segmentation_model}")
        
        # Kiểm tra xem segmentation model có tồn tại không
        if not self.segmentation_model.exists():
            warnings.warn(
                f"Segmentation model not found at {self.segmentation_model}. "
                f"Will use default pretrained model."
            )
            self.use_onnx_segmentation = False
        else:
            self.use_onnx_segmentation = True
            print(f"Using ONNX segmentation model: {self.segmentation_model}")
        
        # Initialize diarization pipeline
        try:
            if self.use_onnx_segmentation:
                # Sử dụng ONNX model
                self.diarization_pipeline = self._init_diarization_with_onnx(
                    embedding_model, use_auth_token
                )
            else:
                # Fallback to default pretrained model
                self.diarization_pipeline = SpeakerDiarization(
                    embedding=embedding_model or {
                        "checkpoint": "pyannote/speaker-diarization-community-1",
                        "subfolder": "embedding"
                    },
                    token=use_auth_token
                )
        except Exception as e:
            print(f"Warning: Could not initialize diarization pipeline: {e}")
            self.diarization_pipeline = None
        
        # Initialize separation pipeline
        try:
            self.separation_pipeline = SpeechSeparation(
                segmentation=separation_model or "pyannote/separation-ami-1.0",
                token=use_auth_token
            )
        except Exception as e:
            print(f"Warning: Could not initialize separation pipeline: {e}")
            self.separation_pipeline = None
    
    def _init_diarization_with_onnx(self, embedding_model, use_auth_token):
        """
        Khởi tạo diarization pipeline với ONNX segmentation model
        
        Sử dụng ONNX model cho segmentation và pretrained model cho embedding
        """
        try:
            # Import ONNX model wrapper
            from diarization_pipeline.onnx_model import load_onnx_segmentation_model
            
            print(f"Loading ONNX segmentation model: {self.segmentation_model}")
            segmentation_inference = load_onnx_segmentation_model(
                self.segmentation_model,
                use_gpu=(self.device.type == "cuda")
            )
            
            # Initialize diarization với custom segmentation model
            return SpeakerDiarization(
                segmentation=segmentation_inference,
                embedding=embedding_model or {
                    "checkpoint": "pyannote/speaker-diarization-community-1",
                    "subfolder": "embedding"
                },
                token=use_auth_token
            )
            
        except Exception as e:
            warnings.warn(
                f"Could not load ONNX model: {e}. "
                f"Falling back to pretrained segmentation model."
            )
            
            # Fallback to default pretrained model
            return SpeakerDiarization(
                embedding=embedding_model or {
                    "checkpoint": "pyannote/speaker-diarization-community-1",
                    "subfolder": "embedding"
                },
                token=use_auth_token
            )
    
    def __call__(
        self,
        audio_file: AudioFile,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> CombinedOutput:
        """
        Áp dụng combined pipeline lên audio file
        
        Parameters
        ----------
        audio_file : AudioFile
            Đường dẫn đến file audio hoặc dict có 'audio' và 'sample_rate'
        num_speakers : int, optional
            Số lượng speakers nếu biết trước
        min_speakers : int, optional
            Số lượng speakers tối thiểu
        max_speakers : int, optional
            Số lượng speakers tối đa
        
        Returns
        -------
        CombinedOutput
            Object chứa diarization timestamps và separated audio
        """
        print(f"\n{'='*60}")
        print(f"Processing: {audio_file}")
        print(f"{'='*60}\n")
        
        # 1. SPEAKER DIARIZATION
        print("Step 1: Running Speaker Diarization...")
        if self.diarization_pipeline is None:
            raise RuntimeError("Diarization pipeline not initialized")
        
        diarization_result = self.diarization_pipeline(
            audio_file,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Handle different output formats
        if hasattr(diarization_result, 'speaker_diarization'):
            diarization = diarization_result.speaker_diarization
        else:
            diarization = diarization_result
        
        # Extract timestamps
        timestamps = self._extract_timestamps(diarization)
        print(f"Found {len(set(t['speaker'] for t in timestamps))} speakers")
        print(f"Total segments: {len(timestamps)}")
        
        # 2. SPEECH SEPARATION
        print("\nStep 2: Running Speech Separation...")
        if self.separation_pipeline is None:
            raise RuntimeError("Separation pipeline not initialized")
        
        separation_result = self.separation_pipeline(
            audio_file,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # 3. COMBINE RESULTS
        print("\nStep 3: Combining Results...")
        separated_sources = self._extract_separated_sources(
            separation_result, 
            diarization,
            audio_file
        )
        
        # Get sample rate
        waveform, sample_rate = self.audio(audio_file)
        
        print("\nProcessing completed!")
        print(f"Separated {len(separated_sources)} speakers")
        
        return CombinedOutput(
            diarization=diarization,
            timestamps=timestamps,
            separated_sources=separated_sources,
            sample_rate=sample_rate
        )
    
    def _extract_timestamps(self, diarization: Annotation) -> List[Dict]:
        """
        Trích xuất timestamps từ Annotation object
        
        Parameters
        ----------
        diarization : Annotation
            Pyannote Annotation object
        
        Returns
        -------
        List[Dict]
            List of {start, end, speaker} dictionaries
        """
        timestamps = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            timestamps.append({
                'start': float(segment.start),
                'end': float(segment.end),
                'duration': float(segment.duration),
                'speaker': speaker
            })
        
        # Sort by start time
        timestamps.sort(key=lambda x: x['start'])
        return timestamps
    
    def _extract_separated_sources(
        self, 
        separation_result,
        diarization: Annotation,
        audio_file: AudioFile
    ) -> Dict[str, np.ndarray]:
        """
        Trích xuất separated audio sources cho mỗi speaker
        
        Parameters
        ----------
        separation_result
            Output từ SpeechSeparation pipeline
        diarization : Annotation
            Diarization annotation
        audio_file : AudioFile
            Original audio file
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping speaker_id to audio array
        """
        # Load original waveform
        waveform, sample_rate = self.audio(audio_file)
        
        separated_sources = {}
        
        # Get unique speakers
        speakers = list(diarization.labels())
        
        # separation_result có thể là Annotation hoặc dict
        # Tùy vào implementation của SpeechSeparation
        if hasattr(separation_result, 'data'):
            # Nếu có separated sources trong result
            for i, speaker in enumerate(speakers):
                # Extract audio cho speaker này
                speaker_audio = self._extract_speaker_audio(
                    waveform, 
                    diarization, 
                    speaker,
                    sample_rate
                )
                separated_sources[speaker] = speaker_audio
        else:
            # Fallback: sử dụng diarization để extract audio
            for speaker in speakers:
                speaker_audio = self._extract_speaker_audio(
                    waveform,
                    diarization,
                    speaker,
                    sample_rate
                )
                separated_sources[speaker] = speaker_audio
        
        return separated_sources
    
    def _extract_speaker_audio(
        self,
        waveform: np.ndarray,
        diarization: Annotation,
        speaker: str,
        sample_rate: int
    ) -> np.ndarray:
        """
        Extract audio segments cho một speaker cụ thể
        
        Parameters
        ----------
        waveform : np.ndarray
            Full audio waveform
        diarization : Annotation
            Diarization annotation
        speaker : str
            Speaker label
        sample_rate : int
            Sample rate of audio
        
        Returns
        -------
        np.ndarray
            Concatenated audio cho speaker
        """
        speaker_segments = []
        
        for segment, _, label in diarization.itertracks(yield_label=True):
            if label == speaker:
                # Convert time to samples
                start_sample = int(segment.start * sample_rate)
                end_sample = int(segment.end * sample_rate)
                
                # Extract segment
                if len(waveform.shape) == 1:
                    segment_audio = waveform[start_sample:end_sample]
                else:
                    segment_audio = waveform[0, start_sample:end_sample]  # mono
                
                speaker_segments.append(segment_audio)
        
        # Concatenate all segments
        if speaker_segments:
            concatenated = np.concatenate(speaker_segments)
        else:
            concatenated = np.array([])
        
        return concatenated


def main():
    """
    Example usage of CombinedPipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combined Speaker Diarization and Speech Separation Pipeline"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "models" / "model.int8.onnx"),
        help="Path to ONNX segmentation model"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers (if known)"
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for separated audio"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for downloading models"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CombinedPipeline(
        segmentation_model=args.segmentation_model,
        use_auth_token=args.hf_token
    )
    
    # Process audio
    result = pipeline(
        args.audio_file,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save separated audio
    print(f"\nSaving separated audio to {output_dir}/")
    result.save_separated_audio(output_dir)
    
    # Save timestamps
    timestamps_json = output_dir / "timestamps.json"
    timestamps_rttm = output_dir / "timestamps.rttm"
    
    result.save_timestamps(timestamps_json)
    result.save_timestamps(timestamps_rttm)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Number of speakers: {len(result.separated_sources)}")
    print(f"Total segments: {len(result.timestamps)}")
    print(f"Sample rate: {result.sample_rate} Hz")
    print("\nTimestamps:")
    for ts in result.timestamps[:10]:  # Print first 10
        print(f"  {ts['start']:.2f}s - {ts['end']:.2f}s: {ts['speaker']}")
    if len(result.timestamps) > 10:
        print(f"  ... and {len(result.timestamps) - 10} more segments")
    print("\nOutput saved to:", output_dir.absolute())
    print("="*60)


if __name__ == "__main__":
    main()
