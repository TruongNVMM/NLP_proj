"""
ONNX Model Wrapper cho Segmentation
Support loading và inference với model.int8.onnx
"""

import warnings
from pathlib import Path
from typing import Union, Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    warnings.warn("onnxruntime not installed. Install with: pip install onnxruntime")

try:
    import torch
    from pyannote.core import SlidingWindowFeature, SlidingWindow
except ImportError:
    raise ImportError("Please install: pip install pyannote.audio torch")


class ONNXSegmentationModel:
    """
    Wrapper cho ONNX segmentation model
    
    Parameters
    ----------
    model_path : str or Path
        Đường dẫn đến ONNX model file (ví dụ: model.int8.onnx)
    use_gpu : bool, optional
        Sử dụng GPU nếu có (default: True)
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        use_gpu: bool = True
    ):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for ONNX models. "
                "Install with: pip install onnxruntime"
            )
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Setup providers (GPU or CPU)
        providers = []
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
            print("Using GPU (CUDA) for ONNX inference")
        else:
            providers.append('CPUExecutionProvider')
            print("Using CPU for ONNX inference")
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        
        # Get model metadata
        self._get_model_info()
        
        print(f"Loaded ONNX model from: {self.model_path}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Output shape: {self.output_shape}")
    
    def _get_model_info(self):
        """Lấy thông tin về input/output của model"""
        # Input info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_dtype = self.session.get_inputs()[0].type
        
        # Output info
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        self.output_dtype = self.session.get_outputs()[0].type
        
        # Determine specifications
        # Typical segmentation model: (batch, channels, samples) -> (batch, frames, speakers)
        self.num_speakers = self.output_shape[-1] if len(self.output_shape) > 2 else None
    
    def __call__(
        self, 
        waveform: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> SlidingWindowFeature:
        """
        Run inference on audio waveform
        
        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor
            Audio waveform with shape (channels, samples) or (samples,)
        
        Returns
        -------
        SlidingWindowFeature
            Segmentation output with speaker probabilities
        """
        # Convert to numpy if tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # Ensure correct shape: (batch, channels, samples)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, np.newaxis, :]  # Add batch and channel dims
        elif waveform.ndim == 2:
            waveform = waveform[np.newaxis, :]  # Add batch dim
        
        # Convert to float32 if needed
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: waveform}
        )
        
        segmentation = outputs[0]  # (batch, frames, speakers)
        
        # Remove batch dimension
        if segmentation.shape[0] == 1:
            segmentation = segmentation[0]  # (frames, speakers)
        
        # Create SlidingWindowFeature
        # Estimate frame duration based on input/output sizes
        num_samples = waveform.shape[-1]
        num_frames = segmentation.shape[0]
        sample_rate = kwargs.get('sample_rate', 16000)
        
        duration = num_samples / sample_rate
        frame_duration = duration / num_frames
        
        sliding_window = SlidingWindow(
            start=0.0,
            duration=frame_duration,
            step=frame_duration
        )
        
        return SlidingWindowFeature(segmentation, sliding_window)
    
    @property
    def specifications(self):
        """
        Return model specifications compatible with pyannote
        """
        class Specs:
            def __init__(self, num_speakers):
                self.classes = [f"speaker_{i}" for i in range(num_speakers)]
        
        if self.num_speakers:
            return Specs(self.num_speakers)
        return None


class ONNXModelInference:
    """
    Inference wrapper tương thích với pyannote.audio.Inference
    
    Parameters
    ----------
    model : ONNXSegmentationModel
        ONNX model instance
    window : str, optional
        Window type for sliding window inference
    duration : float, optional
        Duration of each chunk (seconds)
    step : float, optional
        Step size between chunks (seconds)
    """
    
    def __init__(
        self,
        model: ONNXSegmentationModel,
        window: str = "sliding",
        duration: float = 5.0,
        step: Optional[float] = None,
        batch_size: int = 1,
    ):
        self.model = model
        self.window = window
        self.duration = duration
        self.step = step or (duration * 0.5)  # 50% overlap by default
        self.batch_size = batch_size
        self.sample_rate = 16000
    
    def __call__(self, file_or_waveform, **kwargs):
        """
        Run inference on audio file or waveform
        
        Parameters
        ----------
        file_or_waveform : str, Path, dict, or np.ndarray
            Audio file path or waveform
        
        Returns
        -------
        SlidingWindowFeature
            Segmentation output
        """
        from pyannote.audio import Audio
        
        # Load audio if needed
        if isinstance(file_or_waveform, (str, Path)):
            audio = Audio(sample_rate=self.sample_rate, mono=True)
            waveform, sample_rate = audio(file_or_waveform)
        elif isinstance(file_or_waveform, dict):
            waveform = file_or_waveform['waveform']
            sample_rate = file_or_waveform.get('sample_rate', self.sample_rate)
        else:
            waveform = file_or_waveform
            sample_rate = kwargs.get('sample_rate', self.sample_rate)
        
        # Convert to numpy if tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # Run model
        segmentation = self.model(
            waveform,
            sample_rate=sample_rate,
            **kwargs
        )
        
        return segmentation
    
    @property
    def classes(self):
        """Return class labels"""
        if hasattr(self.model, 'specifications') and self.model.specifications:
            return self.model.specifications.classes
        return []


def load_onnx_segmentation_model(
    model_path: Union[str, Path],
    use_gpu: bool = True,
    duration: float = 5.0,
    step: Optional[float] = None,
) -> ONNXModelInference:
    """
    Helper function để load ONNX segmentation model
    
    Parameters
    ----------
    model_path : str or Path
        Đường dẫn đến ONNX model (ví dụ: model.int8.onnx)
    use_gpu : bool
        Sử dụng GPU nếu có
    duration : float
        Duration của mỗi chunk (seconds)
    step : float, optional
        Step size giữa các chunks (seconds)
    
    Returns
    -------
    ONNXModelInference
        Inference wrapper
    
    Example
    -------
    >>> model = load_onnx_segmentation_model("model.int8.onnx")
    >>> segmentation = model("audio.wav")
    >>> print(segmentation.data.shape)  # (num_frames, num_speakers)
    """
    onnx_model = ONNXSegmentationModel(model_path, use_gpu=use_gpu)
    inference = ONNXModelInference(
        onnx_model,
        duration=duration,
        step=step
    )
    return inference


if __name__ == "__main__":
    """Test ONNX model loading"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ONNX Segmentation Model")
    parser.add_argument("model_path", type=str, help="Path to ONNX model file")
    parser.add_argument("--audio", type=str, help="Audio file to test", default=None)
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    
    args = parser.parse_args()
    
    print("Loading ONNX model...")
    model = load_onnx_segmentation_model(
        args.model_path,
        use_gpu=not args.no_gpu
    )
    
    if args.audio:
        print(f"\nTesting on audio: {args.audio}")
        segmentation = model(args.audio)
        
        print(f"Segmentation shape: {segmentation.data.shape}")
        print(f"Sliding window: {segmentation.sliding_window}")
        print(f"Number of frames: {len(segmentation)}")
        print(f"Classes: {model.classes}")
    else:
        print("\nModel loaded successfully!")
        print("Run with --audio <file> to test inference")
