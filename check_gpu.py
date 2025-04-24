import torch
import platform
import subprocess
import sys
import os

def print_separator():
    print("\n" + "=" * 80 + "\n")

def check_cuda():
    """Comprehensive CUDA and GPU diagnostics"""
    print_separator()
    print("🔍 GPU DIAGNOSTICS REPORT 🔍")
    print_separator()
    
    # System information
    print("SYSTEM INFORMATION:")
    print(f"  OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python: {platform.python_version()} ({sys.executable})")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  PyTorch installed at: {os.path.dirname(torch.__file__)}")
    
    # Check if CUDA is available according to PyTorch
    print_separator()
    print("PYTORCH CUDA STATUS:")
    cuda_available = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {cuda_available}")
    
    if cuda_available:
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        
        # Display information for each GPU
        for i in range(torch.cuda.device_count()):
            print(f"\n  DEVICE {i}:")
            print(f"    Name: {torch.cuda.get_device_name(i)}")
            print(f"    Capability: {torch.cuda.get_device_capability(i)}")
            
            # Memory information
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            print(f"    Total memory: {total_memory:.2f} GB")
    else:
        print("  ⚠️ CUDA is not available in PyTorch!")
        print("  This suggests either:")
        print("    - You have no NVIDIA GPU")
        print("    - Your PyTorch installation doesn't support CUDA")
        print("    - There's a CUDA version mismatch")
    
    # Check for GPU via nvidia-smi
    print_separator()
    print("NVIDIA-SMI STATUS:")
    
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if nvidia_smi.returncode == 0:
            print("  NVIDIA GPU detected by system:")
            print("\n" + nvidia_smi.stdout)
            
            if not cuda_available:
                print("\n  ⚠️ MISMATCH DETECTED: GPU found by system but not by PyTorch!")
                print("  Likely causes:")
                print("    1. PyTorch was installed without CUDA support")
                print("    2. CUDA version mismatch between PyTorch and drivers")
                print("    3. Environment issues (e.g., PATH settings)")
                
                print("\n  SOLUTION:")
                print("    Reinstall PyTorch with CUDA support:")
                if platform.system() == "Windows":
                    print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                else:
                    print("      pip install torch torchvision")
        else:
            print("  nvidia-smi command failed. No NVIDIA driver or GPU detected by system.")
    except Exception as e:
        print(f"  Failed to run nvidia-smi: {str(e)}")
        print("  This suggests NVIDIA drivers are not installed or accessible.")

    # Check for MPS (Apple Silicon)
    print_separator()
    print("APPLE SILICON STATUS:")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  MPS is available - Apple Silicon acceleration can be used")
    else:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print("  ⚠️ Running on Apple Silicon but MPS is not available!")
            print("  Make sure you're using PyTorch 1.12+ with macOS 12.3+")
        else:
            print("  Not running on Apple Silicon")
    
    print_separator()
    print("RECOMMENDATION:")
    if cuda_available:
        print("  ✅ CUDA is working! Use device='cuda' for GPU acceleration.")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  ✅ Use device='mps' for Apple Silicon acceleration.")
    else:
        has_nvidia_gpu = False
        try:
            has_nvidia_gpu = subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
        except:
            pass
            
        if has_nvidia_gpu:
            print("  ⚠️ You have an NVIDIA GPU but PyTorch can't use it!")
            print("  Reinstall PyTorch with the correct CUDA version:")
            print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("  ℹ️ No GPU acceleration available. Using CPU only.")
    
    print_separator()

if __name__ == "__main__":
    check_cuda() 