import os
import sys
import urllib.request
import subprocess
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def download_file(url, dest):
    if not dest.exists():
        print(f"Downloading {url} to {dest}...")
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    else:
        print(f"File {dest} already exists. Skipping download.")

def main():
    # 1. Install requirements
    print("\n--- Installing requirements ---")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", str(ROOT_DIR / "requirements.txt")])

    # 2. Export to ONNX
    print("\n--- Exporting PyTorch model to ONNX ---")
    export_script = ROOT_DIR / "scripts" / "export-onnx.py"
    # Note: export-onnx.py currently expects pytorch_model.bin in its working directory.
    # We will run it from the models directory.
    run_cmd([sys.executable, str(export_script)], cwd=models_dir)

    # 3. Run tests
    print("\n--- Running Tests ---")
    sherpa_utils_dir = ROOT_DIR / "src" / "sherpa_onnx_utils"
    
    # Set PYTHONPATH so python can find the modules
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")

    print("\n[Testing vad-torch.py]")
    vad_torch_script = sherpa_utils_dir / "vad-torch.py"
    subprocess.run([sys.executable, str(vad_torch_script)], cwd=models_dir, env=env)

    print("\n[Testing vad-onnx.py with model.onnx]")
    vad_onnx_script = sherpa_utils_dir / "vad-onnx.py"
    subprocess.run([
        sys.executable, str(vad_onnx_script),
        "--model", str(models_dir / "model.onnx"),
        "--wav", str(test_wav_path)
    ], env=env)

    print("\n[Testing vad-onnx.py with model.int8.onnx]")
    subprocess.run([
        sys.executable, str(vad_onnx_script),
        "--model", str(models_dir / "model.int8.onnx"),
        "--wav", str(test_wav_path)
    ], env=env)

    print("\n--- Setup and Testing Complete! ---")
    print(f"ONNX models are located in: {models_dir}")

if __name__ == "__main__":
    main()
