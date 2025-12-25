# Uma Automation System

A comprehensive automation system with OCR, YOLO object detection, template matching, and screen control.

## Features

- **OCR Text Detection**: Find and read text on screen using EasyOCR
- **YOLO Training System**: Train custom object detectors with interactive Y/N labeling
- **Template Matching**: Find UI elements using OpenCV template matching
- **Screen Capture**: Capture from multiple monitors with live monitoring
- **Automation**: Control mouse and keyboard (PyAutoGUI wrapper)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Install Tesseract OCR:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH

## Quick Start

### Test 1: Find Text on Screen

Run the main test to find specific text on your screen:

```bash
python src/main.py test1
```

This will search for:
- Turn(s) left
- Junior Year Pre-Debut
- Normal, Rest, Training, Skills
- infirmary, Recreation, Races
- Full, Stats, Career, Profile

### Demo All Systems

```bash
python src/main.py demo
```

## Usage Examples

### 1. Screen Capture

```python
from src.screen_capture import ScreenCapture

capture = ScreenCapture()

# Capture primary monitor
screen = capture.capture_monitor(monitor_num=1)

# Capture specific region
region = capture.capture_region(x=100, y=100, width=800, height=600)

# Save screenshot
capture.save_screenshot("screenshot.png")
```

### 2. OCR Text Detection

```python
from src.ocr import extract_text, reader
from PIL import Image
import numpy as np

# Load image
image = Image.open("screen.png")

# Simple text extraction
text = extract_text(image)
print(text)

# Detailed OCR with positions
img_np = np.array(image)
results = reader.readtext(img_np)

for bbox, text, confidence in results:
    print(f"Text: {text}, Position: {bbox}, Confidence: {confidence}")
```

### 3. Template Matching

```python
from src.template_matcher import TemplateMatcher
import cv2

matcher = TemplateMatcher(template_dir="assets")

# Capture screen
screen = cv2.imread("screen.png")

# Find template
match = matcher.find_template(screen, "button_name", threshold=0.8)

if match:
    x, y, w, h, confidence = match
    print(f"Found at ({x}, {y}) with confidence {confidence}")
```

### 4. YOLO Training

```python
from src.yolo_trainer import YOLOTrainer

# Initialize trainer
yolo = YOLOTrainer(project_dir="my_yolo_project")

# Setup classes
yolo.setup_classes(['button', 'icon', 'text_field'])

# Add training images
import cv2
screen = cv2.imread("training_image.jpg")
yolo.add_training_image(screen, "image1.jpg", split='train')

# Interactive labeling with auto-detections
auto_detections = [
    {'class': 0, 'bbox': [100, 100, 50, 30], 'confidence': 0.9}
]
yolo.interactive_label("path/to/image.jpg", auto_detections)

# Train model
yolo.train_model(epochs=50, batch_size=16, model_size='n')

# Use trained model
yolo.load_model()
detections = yolo.predict(screen, confidence=0.5)

for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
```

### 5. Automation

```python
from src.automation import AutomationController, AutomationSequence

auto = AutomationController()

# Simple click
auto.click(100, 200)

# Type text
auto.type_text("Hello World")

# Hotkey
auto.hotkey('ctrl', 'c')

# Create sequence
seq = AutomationSequence()
seq.add_click(100, 200)
seq.add_wait(0.5)
seq.add_type("username")
seq.add_click(100, 250)
seq.add_type("password")
seq.add_hotkey('enter')
seq.execute()
```

### 6. Main Automation Class

```python
from src.main import UmaAutomation

# Initialize
uma = UmaAutomation(monitor_num=1)

# Find text on screen
found = uma.find_text_on_screen(['Training', 'Skills', 'Rest'])

for text, data in found.items():
    print(f"Found '{text}' at {data['position']}")

# Click on text
uma.click_on_text('Training')

# Wait for text to appear
if uma.wait_for_text('Battle Start', timeout=10):
    print("Battle started!")

# Live monitoring
def on_text_found(text, position):
    print(f"Detected {text} at {position}")

uma.live_monitor_text(['Error', 'Success'], callback=on_text_found, fps=2)
```

### 7. Live Monitor

```python
from src.screen_capture import LiveMonitor

def process_frame(frame):
    # Process each frame
    # frame is a PIL Image
    print(f"Processing frame: {frame.size}")

monitor = LiveMonitor(monitor_num=1, fps=10)
monitor.start_monitoring(process_frame)
```

## Roboflow dataset download

Use the included script to download the Roboflow dataset (YOLOv5 format).

Run with your API key:

```bash
python src/download_roboflow_dataset.py --api_key gJQZIVpVxWT4fom1G2Cs
```

Or set the environment variable and run without `--api_key`:

```powershell
set ROBOFLOW_API_KEY=gJQZIVpVxWT4fom1G2Cs
python src/download_roboflow_dataset.py
```

The dataset will be downloaded by the Roboflow SDK into the repository (check the created folder).

## Project Structure

```
Uma/
├── src/
│   ├── main.py              # Main automation system
│   ├── ocr.py               # OCR functions
│   ├── screen_capture.py    # Screen capture system
│   ├── template_matcher.py  # Template matching
│   ├── yolo_trainer.py      # YOLO training system
│   ├── automation.py        # Automation control
│   └── preprocessing.py     # Image preprocessing
├── assets/                  # Template images
│   ├── Buttons/
│   ├── Icons/
│   └── Training/
├── data/                    # Game data
├── debug/                   # Debug output
├── logs/                    # Log files
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Training Custom YOLO Models

1. **Collect Training Images**:
   ```python
   from src.screen_capture import ScreenCapture
   
   capture = ScreenCapture()
   for i in range(10):
       screen = capture.capture_monitor()
       screen.save(f"training_data/img_{i}.jpg")
       time.sleep(2)
   ```

2. **Setup and Label**:
   ```python
   from src.yolo_trainer import YOLOTrainer
   
   yolo = YOLOTrainer()
   yolo.setup_classes(['button', 'icon', 'menu'])
   
   # Add images and label interactively
   # The system will ask Y/N for each detection
   ```

3. **Train**:
   ```python
   yolo.train_model(epochs=100, batch_size=16)
   ```

4. **Use**:
   ```python
   yolo.load_model()
   detections = yolo.predict(screen_image)
   ```

## Tips

- **OCR Accuracy**: Preprocess images (grayscale, threshold) for better results
- **Template Matching**: Store templates in `assets/` folder, organized by category
- **YOLO Training**: Start with 50-100 labeled images per class
- **Automation**: Use `pyautogui.FAILSAFE = True` - move mouse to corner to stop
- **Live Monitoring**: Lower FPS (2-5) to reduce CPU usage

## Troubleshooting

**OCR not finding text:**
- Check image quality and contrast
- Try preprocessing (see preprocessing.py)
- Adjust confidence threshold

**Template matching fails:**
- Ensure template is exact match (same size, colors)
- Try different matching methods
- Check threshold value

**YOLO training errors:**
- Ensure ultralytics and torch are installed
- Check GPU availability with `torch.cuda.is_available()`
- Reduce batch size if running out of memory

## License

MIT
