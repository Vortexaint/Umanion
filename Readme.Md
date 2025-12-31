# Uma Musume Automation System

An intelligent automation system for Uma Musume that plays the game automatically using computer vision and AI. The system can detect game elements, read text, make decisions, and perform actions - handling everything from training selection to race participation.

## 🎮 What Does This Do?

This program **fully automates** the Uma Musume game by:
- **Automatically training your Uma** - Selects the best training options based on stats and energy
- **Managing races** - Schedules and participates in races, changes strategies based on aptitudes
- **Handling events** - Reads event text and selects the best choice from a 40,000+ event database
- **Managing mood and energy** - Uses recreation when mood is low, rests when energy is depleted
- **Reading stats and goals** - Tracks your Uma's progress and adjusts training priorities
- **Recording everything** - Saves videos and logs of the entire automation session

The system uses three detection methods:
1. **YOLO AI** - Fast object detection for buttons and UI elements
2. **OCR** - Reads text for stats, events, and goals
3. **Template Matching** - Precise detection of specific buttons and icons

## 🚀 Quick Start

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the YOLO model:
   - Should be at `yolo_project/train/weights/best.pt`

### Running the Automation

**Basic usage (with video recording):**
```bash
python src/game_automation.py --monitor 2
```

**Without video recording (faster):**
```bash
python src/game_automation.py --monitor 2 --no-video
```

**Stop automation:**
- Press **F1** to gracefully stop the automation

### Monitor Numbers
- `--monitor 1` = Primary monitor
- `--monitor 2` = Secondary monitor (where game is displayed)
- `--monitor 3` = Third monitor, etc.

## 📊 What Happens During Automation?

The automation runs in a continuous loop:

1. **Check Game State** - Captures screen and detects current menu/screen
2. **Check Mood** - Uses recreation if mood is below "Good"
3. **Navigate Menu** - Detects buttons (FullStats, RaceDay, Go, etc.)
4. **Make Decisions**:
   - If race day → Select and participate in race
   - If training available → Choose best training based on stats
   - If event appears → Select optimal choice
   - If energy low → Go to rest
5. **Record Everything** - Saves video and logs all actions

## 🎯 Key Features

- **🤖 Fully Automated**: Runs without any input required
- **🎥 Video Recording**: Records session with detection visualizations (logs/automation_YYYYMMDD_HHMMSS.mp4)
- **📝 Debug Logging**: Comprehensive logs for debugging (logs/automation_YYYYMMDD_HHMMSS.log)
- **⚡ Smart Energy Management**: Automatically rests when energy < 50
- **🏇 Intelligent Race Selection**: Chooses races based on terrain/distance affinity
- **📖 Event Database**: Uses 40,000+ event database to select best choices
- **🎯 Training Optimization**: Analyzes failure rates and stat gains
- **😊 Mood Management**: Maintains good mood automatically

## 📋 System Requirements

- Python 3.8+
- Windows (tested) / Linux / MacOS
- Game displayed on a monitor (monitor 2 recommended)
- YOLO model weights at `yolo_project/train/weights/best.pt`

## 🧪 Testing Components

Before running full automation, you can test individual components:

```bash
# Test screen capture
python src/test_automation.py --test capture --monitor 2

# Test YOLO detection
python src/test_automation.py --test yolo --monitor 2

# Test template matching
python src/test_automation.py --test template --template assets/Buttons/FullStats.png

# Test OCR
python src/test_automation.py --test ocr --region "300,200,600,250"

# Test all components
python src/test_automation.py --test all --monitor 2
```

## 📁 Output Files

### Log Files
- **Location**: `logs/automation_YYYYMMDD_HHMMSS.log`
- **Contents**: 
  - All detections (YOLO, templates, OCR)
  - Decision reasoning
  - Click coordinates
  - Error messages
  - Performance metrics

### Video Recordings
- **Location**: `logs/automation_YYYYMMDD_HHMMSS.mp4`
- **Features**:
  - Visual overlay of all detections
  - YOLO detections shown with bounding boxes
  - Template matches highlighted
  - 10 FPS recording of entire session

### Debug Images
- **Location**: `debug/`
- **Contents**: Event regions, stat crops, and detection screenshots

## 🎮 How It Works

### Detection System
### Debug Images
- **Location**: `debug/`
- **Types**:
  - `test_screenshot.png` - Screen captures
  - `test_yolo_detection.png` - YOLO visualization
  - `test_template_*.png` - Template matches
  - `stat_*.png` - Individual stat crops

## 🎯 Automation Features

### Mood Management
- ✅ Detects mood levels (Awful/Bad/Normal/Good/Great)
- ✅ Automatically uses Recreation when mood < Good
- ✅ Skips mood check during July-August

### Race System
- ✅ Schedules G1 races based on terrain/distance affinity
- ✅ Selects races with highest fans and matching stars
- ✅ Changes strategy based on style aptitude (A/B priority)
- ✅ Handles in-race interactions (Skip, Race2)

### Training Selection
- ✅ Analyzes all training options by hovering
- ✅ Reads failure rate and stat gains for each
- ✅ Selects best training based on current needs
- ✅ Avoids high-failure trainings

### Event Handling
- ✅ Detects event screens
- ✅ Matches events against 40,000+ event database
- ✅ Selects optimal choice based on stat priorities
- ✅ Avoids energy overflow and negative outcomes

### Energy Management
- ✅ Monitors energy levels continuously
- ✅ Goes to rest when energy < 50
- ✅ Prioritizes low-failure training when energy is low
- ✅ Optimizes training selection when energy is high

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
│   ├── game_automation.py      # 🤖 MAIN AUTOMATION SYSTEM
│   ├── test_automation.py      # 🧪 Component testing
│   ├── main.py                 # Legacy automation
│   ├── ocr.py                  # OCR functions
│   ├── screen_capture.py       # Screen capture system
│   ├── template_matcher.py     # Template matching
│   ├── yolo_trainer.py         # YOLO training system
│   ├── automation.py           # Automation control
│   ├── get_stats_region.py     # Stat reading from coordinates
│   ├── stat_treshold.py        # Stat thresholds and checks
│   └── preprocessing.py        # Image preprocessing
├── assets/                     # Template images
│   ├── Buttons/                # Button templates
│   ├── Icons/                  # Icon templates
│   └── Training/               # Training templates
├── data/                       # Game data
│   ├── races.json              # Race definitions
│   ├── events.json             # Event choices database (40K+)
│   └── skills.json             # Skill data
├── yolo_training/              # YOLO dataset
│   ├── data.yaml               # YOLO configuration
│   ├── train/                  # Training images & labels
│   ├── valid/                  # Validation images & labels
│   └── test/                   # Test images & labels
├── yolo_project/               # YOLO outputs
│   └── weights/
│       └── best.pt             # Trained YOLO model
├── debug/                      # Debug output images
├── logs/                       # Log files & video recordings
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── QUICKSTART.md               # 🚀 Quick start guide
└── AUTOMATION_GUIDE.md         # 📚 Full automation documentation
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│           Game Automation System                     │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │  Main Loop (game_automation.py)            │    │
│  │  • Capture Screen                          │    │
│  │  • Check Exit Conditions                   │    │
│  │  • Process Game State                      │    │
│  │  • Execute Actions                         │    │
│  │  • Record Video                            │    │
│  └────────────────────────────────────────────┘    │
│              │        │         │         │         │
│              ▼        ▼         ▼         ▼         │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │
│  │ YOLO │ │ TMpl │ │ OCR  │ │ Auto │ │ Logs │    │
│  │      │ │Match │ │      │ │      │ │      │    │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘    │
│      │        │        │        │        │         │
│      └────────┴────────┴────────┴────────┘         │
│                      │                              │
│                      ▼                              │
│           ┌──────────────────┐                     │
│           │  Decision Engine │                     │
│           │  • Mood Check    │                     │
│           │  • Menu Navigate │                     │
│           │  • Goal Tracking │                     │
│           │  • Race Select   │                     │
│           │  • Training Pick │                     │
│           │  • Event Handle  │                     │
│           └──────────────────┘                     │
│                      │                              │
│                      ▼                              │
│           ┌──────────────────┐                     │
│           │  Game Interface  │                     │
│           │  • Click Buttons │                     │
│           │  • Drag Mouse    │                     │
│           │  • Read Stats    │                     │
│           └──────────────────┘                     │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   Game Screen  │
              └────────────────┘
```

## 📈 Detection Methods

The automation uses three complementary detection methods:

### 1. YOLO Object Detection
**Use Case**: Real-time detection of game elements
- **Pros**: Fast, rotation/scale invariant, handles multiple objects
- **Cons**: Requires training data, can have false positives
- **Examples**: Mood states, buttons, stat indicators

### 2. Template Matching
**Use Case**: Precise button and UI element location
- **Pros**: No training needed, very accurate for exact matches
- **Cons**: Sensitive to scale/rotation, slower
- **Examples**: FullStats button, Close button, specific icons

### 3. OCR Text Reading
**Use Case**: Reading dynamic text and numbers
- **Pros**: Reads any text, understands game state
- **Cons**: Slower, sensitive to font/quality
- **Examples**: Stat values, event text, goal descriptions

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
