"""Main script for Uma automation system.

This script demonstrates all the integrated systems:
- OCR for text detection
- YOLO for object detection
- Template matching for UI elements
- Screen capture from monitors
- Automation control

Test 1: Find specific text on screen
"""

import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path

# Import our modules
from ocr import extract_text, reader
from screen_capture import ScreenCapture, LiveMonitor
from template_matcher import TemplateMatcher
from yolo_trainer import YOLOTrainer
from automation import AutomationController, AutomationSequence
import json
from datetime import datetime
from ocr import extract_number


class UmaAutomation:
    """Main automation system for Uma."""
    
    def __init__(self, monitor_num: int = None):
        """
        Initialize Uma automation system.
        
        Args:
            monitor_num: Monitor to capture (None = auto-detect primary, 1, 2, etc.)
        """
        self.screen_capture = ScreenCapture()
        self.template_matcher = TemplateMatcher(template_dir="assets")
        self.automation = AutomationController()
        
        # Auto-detect primary monitor if not specified
        if monitor_num is None:
            self.monitor_num = self.screen_capture.get_primary_monitor()
            print(f"Auto-detected primary monitor: {self.monitor_num}")
        else:
            self.monitor_num = monitor_num
        
        # YOLO (optional - needs to be trained first)
        self.yolo = None
        
        # Show monitor info
        monitor_info = self.screen_capture.get_monitor_info(self.monitor_num)
        print(f"\nUma Automation initialized")
        print(f"Monitor: {self.monitor_num}")
        print(f"Position: ({monitor_info['left']}, {monitor_info['top']})")
        print(f"Size: {monitor_info['width']}x{monitor_info['height']}")
        print(f"PyAutoGUI Screen size: {self.automation.get_screen_size()}")
        
    def initialize_yolo(self, model_path: str = None):
        """Initialize YOLO detector (if trained)."""
        self.yolo = YOLOTrainer()
        if model_path:
            self.yolo.load_model(model_path)
    
    def find_text_on_screen(self, target_texts: list, debug: bool = False) -> dict:
        """
        Find specific text on screen using OCR.
        
        Args:
            target_texts: List of text strings to find
            debug: Show debug information
            
        Returns:
            Dictionary with found texts and their locations
        """
        # Capture screen
        screen_img = self.screen_capture.capture_monitor(self.monitor_num)
        
        # Convert to numpy array for OCR
        screen_np = np.array(screen_img)
        
        # Run OCR with bounding boxes
        ocr_results = reader.readtext(screen_np)
        
        found_texts = {}
        
        # Check each OCR result against target texts
        for bbox, text, confidence in ocr_results:
            # Normalize text for comparison
            text_normalized = text.lower().strip()
            
            # Check if any target text matches
            for target in target_texts:
                target_normalized = target.lower().strip()
                
                # Check for exact match or contains
                if target_normalized in text_normalized or text_normalized in target_normalized:
                    # Calculate center position
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    center_x = int(sum(x_coords) / len(x_coords))
                    center_y = int(sum(y_coords) / len(y_coords))
                    
                    found_texts[target] = {
                        'text': text,
                        'position': (center_x, center_y),
                        'bbox': bbox,
                        'confidence': confidence
                    }
                    
                    if debug:
                        print(f"✓ Found '{target}' as '{text}' at ({center_x}, {center_y}) [conf: {confidence:.2f}]")
        
        # Report not found
        if debug:
            for target in target_texts:
                if target not in found_texts:
                    print(f"✗ Not found: '{target}'")
        
        return found_texts
    
    def find_text_regions(self, debug: bool = False, save_output: bool = False) -> list:
        """
        Find all text regions on screen.
        
        Args:
            debug: Show debug info
            save_output: Save annotated image
            
        Returns:
            List of all text detections
        """
        screen_img = self.screen_capture.capture_monitor(self.monitor_num)
        screen_np = np.array(screen_img)
        
        # Run OCR
        ocr_results = reader.readtext(screen_np)
        
        if debug:
            print(f"\nFound {len(ocr_results)} text regions:")
            for i, (bbox, text, conf) in enumerate(ocr_results):
                print(f"{i+1}. '{text}' (confidence: {conf:.2f})")
        
        # Visualize if requested
        if save_output:
            vis_img = screen_np.copy()
            for bbox, text, conf in ocr_results:
                # Draw bounding box
                points = np.array(bbox, dtype=np.int32)
                cv2.polylines(vis_img, [points], True, (0, 255, 0), 2)
                
                # Add text label
                x, y = int(bbox[0][0]), int(bbox[0][1])
                cv2.putText(vis_img, text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save
            output_path = "debug/ocr_visualization.jpg"
            Path("debug").mkdir(exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"\nVisualization saved to {output_path}")
        
        return ocr_results
    
    def click_on_text(self, text: str, offset_x: int = 0, offset_y: int = 0) -> bool:
        """
        Find text and click on it.
        
        Args:
            text: Text to find
            offset_x, offset_y: Offset from text center
            
        Returns:
            True if clicked, False if not found
        """
        found = self.find_text_on_screen([text])
        
        if text in found:
            x, y = found[text]['position']
            self.automation.click(x + offset_x, y + offset_y)
            print(f"Clicked on '{text}' at ({x + offset_x}, {y + offset_y})")
            return True
        else:
            print(f"Text '{text}' not found on screen")
            return False
    
    def wait_for_text(self, text: str, timeout: float = 10.0, check_interval: float = 0.5) -> bool:
        """
        Wait for text to appear on screen.
        
        Args:
            text: Text to wait for
            timeout: Maximum wait time (seconds)
            check_interval: Time between checks (seconds)
            
        Returns:
            True if found, False if timeout
        """
        start_time = time.time()
        
        print(f"Waiting for '{text}'...")
        
        while time.time() - start_time < timeout:
            found = self.find_text_on_screen([text])
            if text in found:
                print(f"✓ Found '{text}' after {time.time() - start_time:.1f}s")
                return True
            
            time.sleep(check_interval)
        
        print(f"✗ Timeout waiting for '{text}'")
        return False
    
    def live_monitor_text(self, target_texts: list, callback=None, fps: int = 2):
        """
        Live monitor screen for specific text.
        
        Args:
            target_texts: List of text to monitor
            callback: Function to call when text is found (text, position)
            fps: Frames per second for monitoring
        """
        def process_frame(frame):
            found = self.find_text_on_screen(target_texts)
            
            if found and callback:
                for text, data in found.items():
                    callback(text, data['position'])
        
        monitor = LiveMonitor(self.monitor_num, fps=fps)
        monitor.start_monitoring(process_frame)


def test_1_find_texts(monitor_num=None):
    """
    Test 1: Find specific text on screen
    
    Args:
        monitor_num: Specific monitor to use (None = auto-detect primary)
    
    Target texts:
    - Turn(s) left
    - Junior Year Pre-Debut
    - Normal
    - Rest
    - Training
    - Skills
    - infirmary
    - Recreation
    - Races
    - Full, Stats
    - Career, Profile
    """
    print("=" * 60)
    print("TEST 1: Find Text on Screen")
    print("=" * 60)
    
    # Show available monitors
    from screen_capture import ScreenCapture
    sc = ScreenCapture()
    sc.list_all_monitors()
    
    # Initialize automation (auto-detect primary by default)
    uma = UmaAutomation(monitor_num=monitor_num)
    
    # Target texts to find
    target_texts = [
        "Turn(s) left",
        "Junior Year Pre-Debut",
        "Normal",
        "Rest",
        "Training",
        "Skills",
        "infirmary",
        "Recreation",
        "Races",
        "Full",
        "Stats",
        "Career",
        "Profile"
    ]
    
    print(f"\nSearching for {len(target_texts)} text strings on screen...")
    print("-" * 60)
    
    # Find all text on screen
    found_texts = uma.find_text_on_screen(target_texts, debug=True)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: Found {len(found_texts)}/{len(target_texts)} texts")
    print("=" * 60)
    
    # Show summary
    for target in target_texts:
        if target in found_texts:
            data = found_texts[target]
            print(f"✓ {target:25s} -> '{data['text']}' at {data['position']}")
        else:
            print(f"✗ {target:25s} -> NOT FOUND")
    
    # Save visualization
    print("\n" + "-" * 60)
    print("Generating visualization...")
    uma.find_text_regions(debug=True, save_output=True)
    
    return found_texts


def demo_all_systems(monitor_num=None):
    """Demonstrate all systems.
    
    Args:
        monitor_num: Specific monitor to use (None = auto-detect primary)
    """
    print("\n" + "=" * 60)
    print("UMA AUTOMATION SYSTEM - FULL DEMO")
    print("=" * 60)
    
    uma = UmaAutomation(monitor_num=monitor_num)
    
    # 1. Screen Capture
    print("\n1. SCREEN CAPTURE")
    print("-" * 60)
    screen = uma.screen_capture.capture_monitor(1)
    print(f"Captured screen: {screen.size}")
    
    # 2. OCR
    print("\n2. OCR TEXT DETECTION")
    print("-" * 60)
    texts = uma.find_text_regions(debug=True)
    print(f"Total text regions found: {len(texts)}")
    
    # 3. Template Matching
    print("\n3. TEMPLATE MATCHING")
    print("-" * 60)
    print(f"Loaded templates: {len(uma.template_matcher.templates)}")
    for name in list(uma.template_matcher.templates.keys())[:5]:
        print(f"  - {name}")
    
    # 4. Automation
    print("\n4. AUTOMATION SYSTEM")
    print("-" * 60)
    print(f"Mouse position: {uma.automation.get_mouse_position()}")
    print(f"Screen size: {uma.automation.get_screen_size()}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


def auto_click_loop(text: str = "Training", monitor_num: int = None, interval: float = 2.0, max_iterations: int = 0):
    """
    Continuously scan the screen and click the center of the given text when found.

    Args:
        text: Text to search and click (case-insensitive)
        monitor_num: Monitor to capture (None = auto-detect primary)
        interval: Seconds between scans
        max_iterations: If >0, stop after this many iterations
    """
    uma = UmaAutomation(monitor_num=monitor_num)
    print(f"Starting auto-click loop for '{text}' (interval={interval}s, max_iterations={max_iterations})")
    iteration = 0

    try:
        while True:
            iteration += 1
            found = uma.find_text_on_screen([text])
            if text in found:
                x, y = found[text]['position']
                uma.automation.click(x, y)
                print(f"Clicked '{text}' at ({x}, {y}) (iteration {iteration})")
            else:
                print(f"'{text}' not found (iteration {iteration})")

            if max_iterations and iteration >= max_iterations:
                print("Reached max iterations, stopping.")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("Auto-click loop interrupted by user.")
    except Exception as e:
        print(f"Auto-click loop encountered an error: {e}")


def map_ui_elements(monitor_num: int = None, force: bool = False, save_path: str = "logs/session_state.json") -> dict:
    """
    Scan screen for common UI elements, return mapping and persist to a session log.

    This function will reuse a saved mapping for the same `current_year` unless `force` is True.
    """
    uma = UmaAutomation(monitor_num=monitor_num)

    # Ensure log folder exists
    Path("logs").mkdir(exist_ok=True)

    # Candidates to look for
    targets = [
        "Turn(s) left", "Junior Year Pre-Debut", "Normal", "Rest", "Training", "Skills",
        "infirmary", "Recreation", "Races", "Full", "Stats", "Career", "Profile", "Skip"
    ]

    # Run OCR search
    found = uma.find_text_on_screen(targets)

    # Build mapping
    mapping = {}
    for t in targets:
        if t in found:
            mapping[t] = {
                'text': found[t]['text'],
                'position': found[t]['position'],
                'bbox': found[t]['bbox'],
                'confidence': found[t]['confidence']
            }

    # Determine current year if present
    current_year = None
    if "Junior Year Pre-Debut" in mapping:
        # try to extract a simpler label
        current_year = "Junior Year"
    else:
        # attempt to find any text containing 'Year'
        for k, v in mapping.items():
            if 'year' in v['text'].lower():
                current_year = v['text']
                break

    # Turns left: look near 'Turn(s) left' center and attempt OCR for number at y+60
    turns_left = None
    if "Turn(s) left" in mapping:
        tx, ty = mapping["Turn(s) left"]['position']
        screen = uma.screen_capture.capture_monitor(uma.monitor_num)
        # crop area below the label (x-40..x+40, y+40..y+100)
        left = max(0, tx - 60)
        top = max(0, ty + 40)
        right = left + 120
        bottom = top + 80
        crop = screen.crop((left, top, right, bottom))
        num = extract_number(crop)
        if num >= 0:
            turns_left = num

    # Choose the correct 'Training' if there are duplicates
    training_positions = [v['position'] for k, v in mapping.items() if k.lower().startswith('training') or v['text'].lower().startswith('training')]
    selected_training = None
    if len(training_positions) == 1:
        selected_training = training_positions[0]
    elif len(training_positions) > 1:
        # Heuristics: prefer one containing ':' in OCR text (e.g., 'training:') or the lower one (larger y)
        for k, v in mapping.items():
            if 'training:' in v['text'].lower():
                selected_training = v['position']
                break
        if selected_training is None:
            # choose the one with largest y (assume actionable button lower on UI)
            selected_training = sorted(training_positions, key=lambda p: p[1])[-1]

    ui_state = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'current_year': current_year,
        'turns_left': turns_left,
        'mapping': mapping,
        'selected_training': selected_training
    }

    # Load existing session log
    try:
        if Path(save_path).exists():
            with open(save_path, 'r', encoding='utf-8') as f:
                session_log = json.load(f)
        else:
            session_log = {}
    except Exception:
        session_log = {}

    key = current_year or 'unknown'
    if not force and key in session_log:
        print(f"Found existing mapping for '{key}' in {save_path}, returning saved mapping. Use --force to rescan.")
        return session_log[key]

    # Save mapping under the current year key
    session_log[key] = ui_state
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(session_log, f, indent=2)

    print(f"UI mapping saved to {save_path} under key '{key}'")
    return ui_state


def start_yolo_training(epochs: int = 50, batch: int = 16, img_size: int = 640, project_dir: str = "yolo_project"):
    """Utility to start YOLO training (blocking)."""
    trainer = YOLOTrainer(project_dir=project_dir)
    # default classes; user can update later via trainer.setup_classes([...])
    trainer.setup_classes(['button', 'icon', 'text'])
    trainer.train_model(epochs=epochs, batch_size=batch, img_size=img_size)


if __name__ == "__main__":
    import sys
    
    # Parse monitor number if provided
    monitor = None
    if len(sys.argv) > 2 and sys.argv[2].startswith('--monitor='):
        try:
            monitor = int(sys.argv[2].split('=')[1])
        except ValueError:
            print("Invalid monitor number")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test1":
            test_1_find_texts(monitor_num=monitor)
        elif sys.argv[1] == "demo":
            demo_all_systems(monitor_num=monitor)
        elif sys.argv[1] == "monitors":
            # Show all monitors
            from screen_capture import ScreenCapture
            sc = ScreenCapture()
            sc.list_all_monitors()
        elif sys.argv[1] == "autoclick":
            # Usage: python src/main.py autoclick --text=Training --interval=2.0 --max=0 --monitor=1
            text_arg = "Training"
            interval_arg = 2.0
            max_arg = 0
            monitor_arg = monitor

            for a in sys.argv[2:]:
                if a.startswith('--text='):
                    text_arg = a.split('=', 1)[1]
                elif a.startswith('--interval='):
                    try:
                        interval_arg = float(a.split('=', 1)[1])
                    except ValueError:
                        pass
                elif a.startswith('--max='):
                    try:
                        max_arg = int(a.split('=', 1)[1])
                    except ValueError:
                        pass
                elif a.startswith('--monitor='):
                    try:
                        monitor_arg = int(a.split('=', 1)[1])
                    except ValueError:
                        pass

            auto_click_loop(text=text_arg, monitor_num=monitor_arg, interval=interval_arg, max_iterations=max_arg)
        elif sys.argv[1] == "map-ui":
            # Usage: python src/main.py map-ui [--force] [--save=path] [--monitor=N]
            force_flag = False
            save_arg = "logs/session_state.json"
            monitor_arg = monitor

            for a in sys.argv[2:]:
                if a == '--force':
                    force_flag = True
                elif a.startswith('--save='):
                    save_arg = a.split('=', 1)[1]
                elif a.startswith('--monitor='):
                    try:
                        monitor_arg = int(a.split('=', 1)[1])
                    except ValueError:
                        pass

            state = map_ui_elements(monitor_num=monitor_arg, force=force_flag, save_path=save_arg)
            print('\nUI mapping summary:')
            print(f" Current year: {state.get('current_year')}")
            print(f" Turns left: {state.get('turns_left')}")
            print(f" Selected training: {state.get('selected_training')}")
            print(f" Mapped elements: {', '.join(list(state.get('mapping', {}).keys()))}")

        elif sys.argv[1] == "train-yolo":
            # Usage: python src/main.py train-yolo [--epochs=N] [--batch=N] [--img=SIZE] [--project=dir]
            epochs = 50
            batch = 16
            img_size = 640
            project_dir = 'yolo_project'
            for a in sys.argv[2:]:
                if a.startswith('--epochs='):
                    try:
                        epochs = int(a.split('=', 1)[1])
                    except ValueError:
                        pass
                elif a.startswith('--batch='):
                    try:
                        batch = int(a.split('=', 1)[1])
                    except ValueError:
                        pass
                elif a.startswith('--img='):
                    try:
                        img_size = int(a.split('=', 1)[1])
                    except ValueError:
                        pass
                elif a.startswith('--project='):
                    project_dir = a.split('=', 1)[1]

            start_yolo_training(epochs=epochs, batch=batch, img_size=img_size, project_dir=project_dir)
        else:
            print("Commands:")
            print("  test1 [--monitor=N]  - Run Test 1 (find text)")
            print("  demo [--monitor=N]   - Demo all systems")
            print("  autoclick [--text=TEXT] [--interval=SECONDS] [--max=N] [--monitor=N] - Continuously scan and click TEXT")
            print("  monitors             - List all monitors")
            print("\nExamples:")
            print("  python src/main.py test1")
            print("  python src/main.py test1 --monitor=1")
            print("  python src/main.py monitors")
            print("  python src/main.py autoclick --text=Training --interval=2.0")
    else:
        # Default: Run Test 1 with auto-detect
        test_1_find_texts(monitor_num=monitor)
