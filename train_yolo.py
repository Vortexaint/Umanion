"""
Simple YOLOv5 training script for Windows
"""
import torch
from pathlib import Path

def train_yolo():
    """Train YOLOv5 model with the Roboflow dataset."""
    print("=== YOLOv5 Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Clone YOLOv5 repo if not exists
    yolo_repo = Path("yolov5")
    if not yolo_repo.exists():
        print("\nCloning YOLOv5 repository...")
        import os
        os.system("git clone https://github.com/ultralytics/yolov5")
        print("Installing YOLOv5 requirements...")
        os.system("pip install -r yolov5/requirements.txt")
    
    # Fix data.yaml paths
    data_yaml = Path("yolo_training/data.yaml")
    print(f"\nUsing dataset config: {data_yaml}")
    
    # Read and update paths in data.yaml
    with open(data_yaml, 'r') as f:
        content = f.read()
    
    # Update paths to absolute
    import os
    base_dir = Path.cwd() / "yolo_training"
    updated_content = content.replace(
        "train: Uma-Musume-3/train/images",
        f"train: {base_dir}/train/images"
    ).replace(
        "val: Uma-Musume-3/valid/images",
        f"val: {base_dir}/valid/images"
    ).replace(
        "test: ../test/images",
        f"test: {base_dir}/test/images"
    )
    
    # Write updated data.yaml
    temp_yaml = Path("yolo_training/data_fixed.yaml")
    with open(temp_yaml, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated dataset paths in {temp_yaml}")
    
    # Training parameters
    img_size = 640
    batch_size = 8  # Reduced for Windows compatibility
    epochs = 50
    weights = "yolov5s.pt"  # Start with small pretrained model
    
    print(f"\nTraining configuration:")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Weights: {weights}")
    print(f"  Dataset: {temp_yaml}")
    
    # Run training using subprocess (simpler approach)
    import subprocess
    import sys
    
    # Build training command
    train_cmd = [
        sys.executable,
        "yolov5/train.py",
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", str(temp_yaml),
        "--weights", weights,
        "--project", "yolo_project",
        "--name", "train",
        "--cache"
    ]
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print(f"Command: {' '.join(train_cmd)}\n")
    
    try:
        # Run training
        result = subprocess.run(train_cmd, check=True)
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best weights saved to: yolo_project/train/weights/best.pt")
        print("="*60)
        
        # Copy best.pt to expected location
        import shutil
        weights_dir = Path("yolo_project/weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        trained_weights = Path("yolo_project/train/weights/best.pt")
        if trained_weights.exists():
            target = weights_dir / "best.pt"
            shutil.copy(trained_weights, target)
            print(f"\nCopied best weights to: {target}")
        else:
            print(f"\nWarning: Could not find trained weights at {trained_weights}")
        
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        print("\nTry the ultralytics CLI instead:")
        print(f"  yolo detect train data={temp_yaml} model=yolov5s.pt epochs={epochs} imgsz={img_size} batch={batch_size} project=yolo_project name=train")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*60)
        print("Alternative: Use ultralytics CLI")
        print("="*60)
        print("Try running this command instead:")
        print(f"  yolo detect train data={temp_yaml} model=yolov5s.pt epochs={epochs} imgsz={img_size} batch={batch_size} project=yolo_project name=train")


if __name__ == "__main__":
    train_yolo()
