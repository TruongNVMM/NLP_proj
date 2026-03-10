from typing import Any, Dict

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from pyannote.audio import Model

def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


@torch.no_grad()
def main():
    pt_filename = "./pytorch_model.bin" 
    print(f"Loading model from {pt_filename}...")
    try:
        model = Model.from_pretrained(pt_filename)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'pytorch_model.bin' is present or update 'pt_filename'.")
        return

    model.eval()
    
    # Print specifications for debugging/verification
    print("Model Specifications:")
    print(model.specifications)

    # Basic assertions - adjust based on your specific model if needed
    # Separation models usually accept mono audio
    # model.audio.sample_rate should be 16000 for most pyannote models
    assert model.audio.sample_rate == 16000, f"Unexpected sample rate: {model.audio.sample_rate}"

    # Verify input shape: (batch, channels, samples)
    # Usually 10s chunks for training/inference standard, but can differ.
    # We'll use the model's example_input_array if available, or create a dummy one.
    
    if hasattr(model, "example_input_array") and model.example_input_array is not None:
        dummy_input = model.example_input_array
    else:
        # Create a dummy input: batch=1, channel=1, 10 seconds at 16k
        dummy_input = torch.randn(1, 1, 16000 * 10)

    print(f"Input shape: {dummy_input.shape}")

    # Run inference to check output
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Expected output for separation: (batch, num_sources, samples) or (batch, samples, num_sources)
    # Pyannote 3.0 separation usually outputs (batch, num_sources, num_samples) for waveforms
    
    num_sources = output.shape[1] # Assuming (N, C, T) 
    # If (N, T, C), then output.shape[2]. Adjust logic if needed after seeing output.
    # Note: Modern pyannote.audio separation models (like ConvTasNet) often output (batch, sources, samples).
    
    # We will assume (batch, sources, samples) for 'y' dynamic axes.
    # If the output is (batch, samples, sources), swap 'num_sources' logic.
    
    opset_version = 13
    filename = "model.onnx"
    
    print(f"Exporting to {filename}...")
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        opset_version=opset_version,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 2: "T"}, # (Batch, 1, Time)
            "y": {0: "N", 2: "T"}, # (Batch, Sources, Time) - Change to 1: "T" if output is (Batch, Time, Sources)
        },
    )

    # Metadata creation
    # Receptive field and window size might not be as strictly defined/used as in segmentation 
    # but useful to attempt to retrieve if the model has them.
    
    window_size = "N/A"
    receptive_field_size = "N/A"
    receptive_field_shift = "N/A"

    # Try to get introspection data if available
    try:
        # Often duration/step are in specifications or receptive_field attributes
        if hasattr(model, "specifications") and hasattr(model.specifications, "duration"):
             window_size = int(model.specifications.duration * 16000)
    except:
        pass

    meta_data = {
        "model_type": "pyannote-separation",
        "version": "3.3.2",
        "num_sources": num_sources,
        "sample_rate": model.audio.sample_rate,
        # "window_size": window_size, # Optional, enable if relevant
        "model_author": "pyannote",
        "description": "Exported Pyannote Audio Separation Model",
    }
    
    add_meta_data(filename=filename, meta_data=meta_data)
    print(f"Metadata added: {meta_data}")

    print("Generating int8 quantization models...")
    filename_int8 = "model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        weight_type=QuantType.QUInt8,
    )

    print(f"Saved to {filename} and {filename_int8}")

if __name__ == "__main__":
    main()
