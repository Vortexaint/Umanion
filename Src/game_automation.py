"""
Complete Game Automation System
Handles the full game loop with mood, stats, goals, races, energy, training and events.
"""
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import keyboard
import warnings

# Suppress FutureWarnings from YOLOv5/PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)

from automation import AutomationController
from screen_capture import ScreenCapture
from template_matcher import TemplateMatcher
from yolo_trainer import YOLOTrainer
from ocr import extract_text, extract_number, reader
from get_stats_region import capture_and_read
from stat_treshold import UmaCheck, StyleCondition


class VideoRecorder:
    """Records screen with detection visualizations."""
    
    def __init__(self, output_path: str, fps: int = 10):
        self.output_path = output_path
        self.fps = fps
        self.writer = None
        self.frames = []
        
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the recording."""
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
        
        self.writer.write(frame)
        
    def release(self):
        """Finalize and save the video."""
        if self.writer:
            self.writer.release()


class GameAutomation:
    """Main automation system for the game."""
    
    def __init__(self, model_path: str = "yolo_project/weights/best.pt", 
                 monitor: int = 1, record_video: bool = True):
        """
        Initialize the game automation system.
        
        Args:
            model_path: Path to YOLO model weights
            monitor: Monitor number to capture from
            record_video: Whether to record video with visualizations
        """
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.automation = AutomationController()
        self.screen_capture = ScreenCapture()
        self.template_matcher = TemplateMatcher("assets")
        self.yolo = YOLOTrainer()
        
        # Load YOLO model
        self.logger.info(f"Loading YOLO model from {model_path}")
        self.yolo.load_model(model_path)
        
        # Load YOLO class names from data.yaml
        data_yaml_path = Path("yolo_training/data.yaml")
        if data_yaml_path.exists():
            import yaml
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                self.yolo.setup_classes(data.get('names', []))
        
        self.monitor = monitor
        self.record_video = record_video
        self.video_recorder = None
        
        # Load game data
        self.load_game_data()
        
        # State tracking
        self.uma_stats = {}  # Track, Distance, Style affinities
        self.current_stats = {}  # Speed, Stamina, Power, Guts, Wit
        self.fullstats_done = False
        self.race_schedule = []  # List of scheduled races
        self.current_month = None
        self.consecutive_event_detections = 0  # Track consecutive event detection loops
        self.stop_requested = False  # F1 key pressed flag
        
        # Setup F1 hotkey to stop automation
        keyboard.add_hotkey('f1', self.request_stop)
        self.logger.info("Press F1 to stop automation gracefully")
    
    def setup_logging(self):
        """Configure logging system."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"automation_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GameAutomation")
        self.logger.info("="*60)
        self.logger.info("Game Automation System Started")
        self.logger.info("="*60)
        
    def load_game_data(self):
        """Load races and events data from JSON files."""
        self.logger.info("Loading game data...")
        
        races_path = Path("data/races.json")
        events_path = Path("data/events.json")
        
        if races_path.exists():
            with open(races_path, 'r', encoding='utf-8') as f:
                self.races_data = json.load(f)
            self.logger.info(f"Loaded races data")
        else:
            self.races_data = {}
            self.logger.warning("races.json not found")
            
        if events_path.exists():
            with open(events_path, 'r', encoding='utf-8') as f:
                self.events_data = json.load(f)
            self.logger.info(f"Loaded events data")
        else:
            self.events_data = {}
            self.logger.warning("events.json not found")
    
    def capture_and_visualize(self) -> Tuple[Image.Image, np.ndarray]:
        """Capture screen and prepare for visualization."""
        screenshot = self.screen_capture.capture_monitor(self.monitor)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Record every frame to video
        if self.video_recorder:
            self.video_recorder.add_frame(frame)
        
        return screenshot, frame
    
    def detect_yolo(self, screenshot: Image.Image, labels: List[str] = None,
                   confidence: float = 0.25) -> Dict[str, List[dict]]:
        """
        Run YOLO detection and return results grouped by label.
        
        Args:
            screenshot: PIL Image to detect on
            labels: Optional list of labels to filter for
            confidence: Minimum confidence threshold
            
        Returns:
            Dictionary mapping label names to list of detections
        """
        detections = self.yolo.predict(screenshot, conf=confidence)
        
        # Group by label
        results = {}
        for det in detections:
            label = det['label']
            if labels is None or label in labels:
                if label not in results:
                    results[label] = []
                results[label].append(det)
        
        return results
    
    def visualize_detections(self, frame: np.ndarray, detections: List[dict],
                           color: Tuple[int, int, int] = (0, 255, 0),
                           label_text: str = None) -> np.ndarray:
        """Draw bounding boxes on frame."""
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            text = label_text or f"{det['label']} {det['conf']:.2f}"
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def find_template(self, screenshot: Image.Image, template_path: str,
                     threshold: float = 0.8) -> Optional[Tuple[int, int]]:
        """Find template in screenshot using template matching."""
        template_name = Path(template_path).stem
        screen_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        result = self.template_matcher.find_template(
            screen_cv, template_name, threshold=threshold
        )
        
        if result:
            # result is a tuple: (x, y, w, h, confidence)
            x, y, w, h, confidence = result
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y)
        return None
    
    def click_detection(self, detection: dict, offset_x: int = 0, offset_y: int = 0):
        """Click on a YOLO detection box center."""
        x1, y1, x2, y2 = detection['box']
        center_x = (x1 + x2) // 2 + offset_x
        center_y = (y1 + y2) // 2 + offset_y
        
        self.logger.info(f"Clicking {detection['label']} at ({center_x}, {center_y})")
        self.automation.click(center_x, center_y)
        time.sleep(0.5)
    
    def click_template(self, pos: Tuple[int, int], label: str = ""):
        """Click on template match position."""
        self.logger.info(f"Clicking {label} at {pos}")
        self.automation.click(pos[0], pos[1])
        time.sleep(0.5)
    
    # ========== MAIN LOOP ==========
    
    def request_stop(self):
        """Request automation stop via F1 key."""
        self.logger.info("F1 pressed - requesting stop")
        self.stop_requested = True
    
    def run(self):
        """Main automation loop."""
        self.logger.info("Starting main automation loop...")
        
        # Move cursor to (50, 50) to avoid blocking UI elements (not 0,0 to avoid PyAutoGUI fail-safe)
        self.automation.move_to(50, 50)
        time.sleep(0.1)
        
        # Setup video recording
        if self.record_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"logs/automation_{timestamp}.mp4"
            self.video_recorder = VideoRecorder(video_path, fps=10)
            self.logger.info(f"Recording to {video_path}")
        
        try:
            loop_count = 0
            while True:
                # Check if F1 was pressed
                if self.stop_requested:
                    self.logger.info("Stop requested via F1 - exiting automation")
                    break
                
                loop_count += 1
                self.logger.info(f"\n{'='*60}\nLoop iteration {loop_count}\n{'='*60}")
                
                # Capture screen
                screenshot, frame = self.capture_and_visualize()
                
                # Check exit conditions
                if self.check_exit_conditions(screenshot, frame):
                    break
                
                # Check month and mood
                if not self.check_month_and_mood(screenshot, frame):
                    continue
                
                # Check for lingering Next buttons before menu
                self.check_and_click_next(screenshot, frame)
                
                # Check for ChangeStrategy button (in-race scenario)
                if self.check_change_strategy(screenshot, frame):
                    self.race_function(screenshot, frame)
                    continue
                
                # Main menu navigation
                self.menu_function(screenshot, frame)
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            self.logger.info("Automation stopped by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def check_exit_conditions(self, screenshot: Image.Image, frame: np.ndarray) -> bool:
        """Check if we should stop the automation."""
        # Check for CompleteCareer button
        pos = self.find_template(screenshot, "assets/Buttons/CompleteCareer.png", threshold=0.8)
        if pos:
            self.logger.info("CompleteCareer button detected - stopping automation")
            cv2.putText(frame, "COMPLETE CAREER DETECTED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            return True
        
        # Check for alarmclocks (YOLO)
        detections = self.detect_yolo(screenshot, labels=['alarmclocks'])
        if 'alarmclocks' in detections:
            self.logger.error("Alarmclocks detected - failed run")
            cv2.putText(frame, "FAILED RUN - ALARMCLOCKS", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            self.visualize_detections(frame, detections['alarmclocks'], (0, 0, 255))
            return True
        
        return False
    
    def check_month_and_mood(self, screenshot: Image.Image, frame: np.ndarray) -> bool:
        """Check month and mood, handle recreation if needed."""
        # Try to detect current month from screen
        # TODO: Implement proper month detection via OCR
        # For now, check if we have it stored
        
        # Check if it's July-August (skip mood check)
        if self.current_month and self.current_month in ['Early Jul', 'Late Jul', 'Early Aug', 'Late Aug']:
            self.logger.info(f"Month is {self.current_month} - skipping mood check")
            return True
        
        # Always check mood if not in July-August
        return self.check_mood(screenshot, frame)
    
    def check_mood(self, screenshot: Image.Image, frame: np.ndarray) -> bool:
        """
        Check mood and handle recreation if needed.
        Returns True if mood is good enough to continue.
        """
        # Detect mood (Awful < Bad < Normal < Good < Great)
        detections = self.detect_yolo(screenshot, 
                                     labels=['awful', 'bad', 'normal', 'good', 'Great'])
        
        mood_hierarchy = {'awful': 0, 'bad': 1, 'normal': 2, 'good': 3, 'Great': 4}
        current_mood = None
        mood_value = -1
        
        for mood_name, dets in detections.items():
            if dets:
                current_mood = mood_name
                mood_value = mood_hierarchy.get(mood_name, -1)
                self.visualize_detections(frame, dets, (255, 255, 0), f"Mood: {mood_name}")
                break
        
        if current_mood:
            self.logger.info(f"Current mood: {current_mood} (value: {mood_value})")
        
        # If mood >= Good (3), we're good
        if mood_value >= 3:
            self.logger.info("Mood is Good or Great - continuing")
            return True
        
        # If mood < Good, use recreation
        if mood_value >= 0 and mood_value < 3:
            self.logger.info("Mood is below Good - using recreation")
            self.use_recreation(screenshot, frame)
            return False  # Need to recheck
        
        # No mood detected, assume we need to continue
        return True
    
    def check_and_click_next(self, screenshot: Image.Image, frame: np.ndarray):
        """Check for lingering Next buttons and click them before proceeding to menu."""
        max_attempts = 3
        for attempt in range(max_attempts):
            screenshot, frame = self.capture_and_visualize()
            
            # Check for Next button via YOLO
            next_detections = self.detect_yolo(screenshot, labels=['Next'])
            if 'Next' in next_detections and next_detections['Next']:
                self.logger.info(f"Lingering Next button detected (YOLO) - clicking (attempt {attempt+1})")
                self.visualize_detections(frame, next_detections['Next'], (100, 255, 100))
                self.click_detection(next_detections['Next'][0])
                time.sleep(0.5)
                continue
            
            # Check for Next button via template
            pos = self.find_template(screenshot, "assets/Buttons/Next.png", threshold=0.8)
            if pos:
                self.logger.info(f"Lingering Next button found (template) - clicking (attempt {attempt+1})")
                cv2.circle(frame, pos, 10, (100, 255, 100), -1)
                self.click_template(pos, "Next")
                time.sleep(0.5)
                continue
            
            # Check for Next button via OCR in region (433, 911 to 672, 970)
            next_region = screenshot.crop((433, 911, 672, 970))
            next_region_np = np.array(next_region)
            
            try:
                results = reader.readtext(next_region_np)
                for (bbox, text, confidence) in results:
                    if "next" in text.strip().lower():
                        # Calculate center of detected text
                        adjusted_x = int(bbox[0][0] + bbox[2][0]) // 2 + 433
                        adjusted_y = int(bbox[0][1] + bbox[2][1]) // 2 + 911
                        self.logger.info(f"Lingering Next button found (OCR) at ({adjusted_x}, {adjusted_y}) - clicking (attempt {attempt+1})")
                        cv2.circle(frame, (adjusted_x, adjusted_y), 10, (100, 255, 100), -1)
                        self.automation.click(adjusted_x, adjusted_y)
                        time.sleep(0.5)
                        break
                else:
                    # No OCR match, try coordinate fallback
                    # Check if there's any visible content in the region
                    continue
            except Exception as e:
                self.logger.debug(f"OCR Next detection error: {e}")
            
            # No Next button found - exit loop
            break
    
    def check_change_strategy(self, screenshot: Image.Image, frame: np.ndarray) -> bool:
        """Check if ChangeStrategy button is present (in-race scenario)."""
        # Method 1: Template matching
        pos = self.find_template(screenshot, "assets/Buttons/ChangeStrategy.png", threshold=0.8)
        if pos:
            self.logger.info(f"ChangeStrategy button detected (template) at {pos} - continuing with race function")
            cv2.circle(frame, pos, 10, (255, 128, 0), -1)
            return True
        
        # Method 2: OCR looking for "Change" text
        try:
            screenshot_np = np.array(screenshot)
            results = reader.readtext(screenshot_np)
            for (bbox, text, confidence) in results:
                if "change" in text.strip().lower():
                    adjusted_x = int(bbox[0][0] + bbox[2][0]) // 2
                    adjusted_y = int(bbox[0][1] + bbox[2][1]) // 2
                    self.logger.info(f"ChangeStrategy detected via OCR at ({adjusted_x}, {adjusted_y}): '{text}' - continuing with race function")
                    cv2.circle(frame, (adjusted_x, adjusted_y), 10, (255, 128, 0), -1)
                    return True
        except Exception as e:
            self.logger.debug(f"OCR ChangeStrategy detection error: {e}")
        
        return False
    
    def use_recreation(self, screenshot: Image.Image, frame: np.ndarray):
        """Click recreation button to improve mood."""
        # Try YOLO detection first
        detections = self.detect_yolo(screenshot, labels=['recreation'])
        if 'recreation' in detections and detections['recreation']:
            self.logger.info("Recreation detected via YOLO")
            self.visualize_detections(frame, detections['recreation'], (0, 255, 255))
            self.click_detection(detections['recreation'][0])
            time.sleep(3)
            return
        
        # Try template matching
        pos = self.find_template(screenshot, "assets/Buttons/Recreation.png", threshold=0.8)
        if pos:
            self.logger.info("Recreation detected via template matching")
            cv2.circle(frame, pos, 10, (0, 255, 255), -1)
            self.click_template(pos, "Recreation")
            time.sleep(3)
            return
        
        self.logger.warning("Recreation button not found")
    
    def menu_function(self, screenshot: Image.Image, frame: np.ndarray):
        """Main menu navigation logic."""
        self.logger.info("=== Menu Function ===")
        
        # Priority 1: Check for events (they block other UI)
        if self.check_for_event(screenshot, frame):
            self.consecutive_event_detections += 1
            self.handle_event(screenshot, frame)
            return
        else:
            # Reset counter when no event detected
            if self.consecutive_event_detections > 0:
                self.logger.debug(f"No event detected - resetting counter from {self.consecutive_event_detections}")
            self.consecutive_event_detections = 0
        
        # Priority 2: Check for FullStats button if not done yet
        if not self.fullstats_done:
            pos = self.find_template(screenshot, "assets/Buttons/FullStats.png", threshold=0.8)
            if pos:
                self.logger.info("FullStats button found - getting necessary stats")
                cv2.circle(frame, pos, 10, (255, 0, 255), -1)
                self.click_template(pos, "FullStats")
                time.sleep(1)
                self.get_necessary_stats(screenshot, frame)
                return
            
            # If FullStats not detected, pass to Event check (already done above)
            # Then check for Infirmary
            detections = self.detect_yolo(screenshot, labels=['infirmary'])
            if 'infirmary' in detections and detections['infirmary']:
                self.logger.info("Infirmary detected - clicking")
                self.visualize_detections(frame, detections['infirmary'], (255, 100, 0))
                self.click_detection(detections['infirmary'][0])
                time.sleep(1)
                return
            
            # If no event or infirmary, check race_day
            race_day_detections = self.detect_yolo(screenshot, labels=['race_day'])
            if 'race_day' in race_day_detections and race_day_detections['race_day']:
                self.logger.info("Race day detected (YOLO)")
                self.visualize_detections(frame, race_day_detections['race_day'], (0, 255, 100))
                self.click_detection(race_day_detections['race_day'][0])
                time.sleep(1)
                self.handle_race_entry(screenshot, frame)
                return
            
            # Try template matching for race_day
            race_day_pos = self.find_template(screenshot, "assets/Buttons/RaceDay.png", threshold=0.8)
            if race_day_pos:
                self.logger.info("Race day detected (template)")
                cv2.circle(frame, race_day_pos, 10, (0, 255, 100), -1)
                self.click_template(race_day_pos, "RaceDay")
                time.sleep(1)
                self.handle_race_entry(screenshot, frame)
                return
            
            # Try template matching for race_day URA variant
            race_day_ura_pos = self.find_template(screenshot, "assets/Buttons/RaceDayUra.png", threshold=0.8)
            if race_day_ura_pos:
                self.logger.info("Race day URA detected (template)")
                cv2.circle(frame, race_day_ura_pos, 10, (0, 255, 100), -1)
                self.click_template(race_day_ura_pos, "RaceDayUra")
                time.sleep(1)
                self.handle_race_entry(screenshot, frame)
                return
            
            # Try OCR for race_day URA variant in region (600, 630 to 762, 956)
            race_day_ura_region = screenshot.crop((600, 630, 762, 956))
            race_day_ura_region_np = np.array(race_day_ura_region)
            
            try:
                results = reader.readtext(race_day_ura_region_np)
                for (bbox, text, confidence) in results:
                    if "ura" in text.strip().lower():
                        adjusted_x = int(bbox[0][0] + bbox[2][0]) // 2 + 600
                        adjusted_y = int(bbox[0][1] + bbox[2][1]) // 2 + 630
                        self.logger.info(f"Race day URA detected via OCR at ({adjusted_x}, {adjusted_y}): '{text}'")
                        cv2.circle(frame, (adjusted_x, adjusted_y), 10, (0, 255, 100), -1)
                        self.automation.click(adjusted_x, adjusted_y)
                        time.sleep(1)
                        self.handle_race_entry(screenshot, frame)
                        return
            except Exception as e:
                self.logger.debug(f"OCR race_day URA detection error: {e}")
            
            # If nothing detected, check for Go buttons
            go_detections = self.detect_yolo(screenshot, labels=['go'])
            if 'go' in go_detections and go_detections['go']:
                self.logger.info("Go button detected (YOLO)")
                self.visualize_detections(frame, go_detections['go'], (100, 255, 100))
                self.click_detection(go_detections['go'][0])
                time.sleep(1)
                return
            
            # Try template matching for Go1/Go2
            for go_btn in ['Go1', 'Go2']:
                pos = self.find_template(screenshot, f"assets/Buttons/{go_btn}.png", threshold=0.8)
                if pos:
                    self.logger.info(f"{go_btn} button found")
                    cv2.circle(frame, pos, 10, (100, 255, 100), -1)
                    self.click_template(pos, go_btn)
                    time.sleep(1)
                    return
        
        # Priority 3: If fullstats done, check current stats first
        if self.fullstats_done:
            self.check_current_stats(screenshot, frame)
            
            # Check for infirmary
            detections = self.detect_yolo(screenshot, labels=['infirmary'])
            if 'infirmary' in detections and detections['infirmary']:
                self.logger.info("Infirmary detected - clicking")
                self.visualize_detections(frame, detections['infirmary'], (255, 100, 0))
                self.click_detection(detections['infirmary'][0])
                time.sleep(1)
                return
            
            # Call goal function
            self.goal_function(screenshot, frame)
            return
        
        self.logger.warning("No actionable element found in menu")
    
    def get_necessary_stats(self, screenshot: Image.Image, frame: np.ndarray):
        """Get Track, Distance, and Style affinities."""
        self.logger.info("=== Getting Necessary Stats ===")
        
        time.sleep(1)  # Wait for stats screen to load
        screenshot, frame = self.capture_and_visualize()
        
        # TODO: Implement actual stat reading from the FullStats screen
        # This would involve OCR on specific regions to extract:
        # - Track affinity (Turf: A/B/C/D/E/F/G, Dirt: A/B/C/D/E/F/G)
        # - Distance affinity (Sprint/Mile/Medium/Long: A/B/C/D/E/F/G)
        # - Style affinity (Front/Pace/Late/End: A/B/C/D/E/F/G)
        
        # Placeholder: Store dummy data
        self.uma_stats = {
            'track_affinity': {'Turf': 'A', 'Dirt': 'B'},
            'distance_affinity': {'Sprint': 'B', 'Mile': 'A', 'Medium': 'A', 'Long': 'C'},
            'style_affinity': {'Front': 'B', 'Pace': 'A', 'Late': 'A', 'End': 'C'}
        }
        
        self.logger.info(f"Uma stats captured: {self.uma_stats}")
        
        # Find and click Close button
        pos = self.find_template(screenshot, "assets/Buttons/Close.png", threshold=0.8)
        if pos:
            self.logger.info("Close button found")
            cv2.circle(frame, pos, 10, (0, 100, 255), -1)
            self.click_template(pos, "Close")
            self.fullstats_done = True
            time.sleep(1)
        else:
            self.logger.warning("Close button not found")
    
    def check_current_stats(self, screenshot: Image.Image, frame: np.ndarray):
        """Check current Speed/Stamina/Power/Guts/Wit stats."""
        self.logger.info("=== Checking Current Stats ===")
        
        # Use YOLO to detect cur_stats region
        detections = self.detect_yolo(screenshot, labels=['cur_stats'])
        if 'cur_stats' in detections and detections['cur_stats']:
            self.visualize_detections(frame, detections['cur_stats'], (255, 0, 255))
        
        # Use coordinate-based capture from get_stats_region.py
        # Returns a dict: {'speed': 148, 'stamina': 115, 'power': 209, 'guts': 95, 'wit': 134}
        try:
            self.current_stats = capture_and_read(307, 722, 745, 745, monitor=self.monitor)
            self.logger.info(f"Current stats: {self.current_stats}")
        except Exception as e:
            self.logger.error(f"Error reading stats: {e}")
            self.current_stats = {'speed': -1, 'stamina': -1, 'power': -1, 'guts': -1, 'wit': -1}
    
    def goal_function(self, screenshot: Image.Image, frame: np.ndarray):
        """Handle goal checking and race scheduling."""
        self.logger.info("=== Goal Function ===")
        
        time.sleep(1)
        screenshot, frame = self.capture_and_visualize()
        
        # Detect goal box
        detections = self.detect_yolo(screenshot, labels=['goal'])
        if 'goal' not in detections or not detections['goal']:
            self.logger.warning("Goal box not detected - checking race_day, event, infirmary, Go, then stats")
            
            # 1. Check for race_day (YOLO detection)
            race_day_detections = self.detect_yolo(screenshot, labels=['race_day'])
            if 'race_day' in race_day_detections and race_day_detections['race_day']:
                self.logger.info("Race day detected from goal function (no goal box) - YOLO")
                self.visualize_detections(frame, race_day_detections['race_day'], (0, 255, 100))
                self.click_detection(race_day_detections['race_day'][0])
                time.sleep(1)
                self.handle_race_entry(screenshot, frame)
                return
            
            # Try template matching for race_day
            race_day_pos = self.find_template(screenshot, "assets/Buttons/RaceDay.png", threshold=0.8)
            if race_day_pos:
                self.logger.info("Race day detected from goal function (no goal box) - template")
                cv2.circle(frame, race_day_pos, 10, (0, 255, 100), -1)
                self.click_template(race_day_pos, "RaceDay")
                time.sleep(1)
                self.handle_race_entry(screenshot, frame)
                return
            
            # Try template matching for race_day URA variant
            race_day_ura_pos = self.find_template(screenshot, "assets/Buttons/RaceDayUra.png", threshold=0.8)
            if race_day_ura_pos:
                self.logger.info("Race day URA detected from goal function (no goal box) - template")
                cv2.circle(frame, race_day_ura_pos, 10, (0, 255, 100), -1)
                self.click_template(race_day_ura_pos, "RaceDayUra")
                time.sleep(1)
                self.handle_race_entry(screenshot, frame)
                return
            
            # Try OCR for race_day URA variant in region (600, 630 to 762, 956)
            race_day_ura_region = screenshot.crop((600, 630, 762, 956))
            race_day_ura_region_np = np.array(race_day_ura_region)
            
            try:
                results = reader.readtext(race_day_ura_region_np)
                for (bbox, text, confidence) in results:
                    if "ura" in text.strip().lower():
                        adjusted_x = int(bbox[0][0] + bbox[2][0]) // 2 + 600
                        adjusted_y = int(bbox[0][1] + bbox[2][1]) // 2 + 630
                        self.logger.info(f"Race day URA detected from goal function via OCR at ({adjusted_x}, {adjusted_y}): '{text}'")
                        cv2.circle(frame, (adjusted_x, adjusted_y), 10, (0, 255, 100), -1)
                        self.automation.click(adjusted_x, adjusted_y)
                        time.sleep(1)
                        self.handle_race_entry(screenshot, frame)
                        return
            except Exception as e:
                self.logger.debug(f"OCR race_day URA detection error: {e}")
            
            # 2. Check for Event
            if self.check_for_event(screenshot, frame):
                self.logger.info("Event detected from goal function (no goal box)")
                self.handle_event(screenshot, frame)
                return
            
            # 3. Check for infirmary
            infirmary_detections = self.detect_yolo(screenshot, labels=['infirmary'])
            if 'infirmary' in infirmary_detections and infirmary_detections['infirmary']:
                self.logger.info("Infirmary detected from goal function (no goal box)")
                self.visualize_detections(frame, infirmary_detections['infirmary'], (255, 100, 0))
                self.click_detection(infirmary_detections['infirmary'][0])
                time.sleep(1)
                return
            
            # 4. Check for Go buttons
            go_detections = self.detect_yolo(screenshot, labels=['go'])
            if 'go' in go_detections and go_detections['go']:
                self.logger.info("Go button detected (YOLO) from goal function (no goal box)")
                self.visualize_detections(frame, go_detections['go'], (100, 255, 100))
                self.click_detection(go_detections['go'][0])
                time.sleep(1)
                return
            
            # Try template matching for Go1/Go2
            for go_btn in ['Go1', 'Go2']:
                pos = self.find_template(screenshot, f"assets/Buttons/{go_btn}.png", threshold=0.8)
                if pos:
                    self.logger.info(f"{go_btn} button found from goal function (no goal box)")
                    cv2.circle(frame, pos, 10, (100, 255, 100), -1)
                    self.click_template(pos, go_btn)
                    time.sleep(1)
                    return
            
            # 5. No race_day/event/infirmary/Go detected, check stats then energy
            self.logger.info("No race_day/event/infirmary/Go detected - checking stats")
            self.check_current_stats(screenshot, frame)
            self.energy_function(screenshot, frame)
            return
        
        goal_det = detections['goal'][0]
        self.visualize_detections(frame, [goal_det], (255, 255, 0))
        x1, y1, x2, y2 = goal_det['box']
        
        # Crop goal region for OCR
        goal_region = screenshot.crop((x1, y1, x2, y2))
        goal_text = extract_text(goal_region)
        self.logger.info(f"Goal text: {goal_text}")
        
        # Check if goal achieved
        if "goal achived" in goal_text.lower() or "goal achieved" in goal_text.lower():
            self.logger.info("Goal achieved!")
            self.energy_function(screenshot, frame)
            return
        
        # Check for turns
        turn_detections = self.detect_yolo(screenshot, labels=['turns'])
        if 'turns' in turn_detections and turn_detections['turns']:
            turn_det = turn_detections['turns'][0]
            tx1, ty1, tx2, ty2 = turn_det['box']
            turn_region = screenshot.crop((tx1, ty1, tx2, ty2))
            turns_remaining = extract_number(turn_region)
            self.logger.info(f"Turns remaining: {turns_remaining}")
            
            # If turns < 6 and goal has "earn", check for race
            if turns_remaining < 6 and "earn" in goal_text.lower():
                self.handle_fan_race(screenshot, frame)
                return
        
        # Check for "In G1" goal
        if "in g1" in goal_text.lower():
            self.handle_g1_race_schedule(screenshot, frame)
            return
        
        # Check for race_day (YOLO detection)
        race_day_detections = self.detect_yolo(screenshot, labels=['race_day'])
        if 'race_day' in race_day_detections and race_day_detections['race_day']:
            self.logger.info("Race day detected from goal function (YOLO)")
            self.visualize_detections(frame, race_day_detections['race_day'], (0, 255, 100))
            self.click_detection(race_day_detections['race_day'][0])
            time.sleep(1)
            self.handle_race_entry(screenshot, frame)
            return
        
        # Try template matching for race_day
        race_day_pos = self.find_template(screenshot, "assets/Buttons/RaceDay.png", threshold=0.8)
        if race_day_pos:
            self.logger.info("Race day detected from goal function (template)")
            cv2.circle(frame, race_day_pos, 10, (0, 255, 100), -1)
            self.click_template(race_day_pos, "RaceDay")
            time.sleep(1)
            self.handle_race_entry(screenshot, frame)
            return
        
        # Check for Event
        if self.check_for_event(screenshot, frame):
            self.logger.info("Event detected from goal function")
            self.handle_event(screenshot, frame)
            return
        
        # Check for infirmary
        infirmary_detections = self.detect_yolo(screenshot, labels=['infirmary'])
        if 'infirmary' in infirmary_detections and infirmary_detections['infirmary']:
            self.logger.info("Infirmary detected from goal function")
            self.visualize_detections(frame, infirmary_detections['infirmary'], (255, 100, 0))
            self.click_detection(infirmary_detections['infirmary'][0])
            time.sleep(1)
            return
        
        # If nothing detected, check for Go buttons
        go_detections = self.detect_yolo(screenshot, labels=['go'])
        if 'go' in go_detections and go_detections['go']:
            self.logger.info("Go button detected (YOLO) from goal function")
            self.visualize_detections(frame, go_detections['go'], (100, 255, 100))
            self.click_detection(go_detections['go'][0])
            time.sleep(1)
            return
        
        # Try template matching for Go1/Go2
        for go_btn in ['Go1', 'Go2']:
            pos = self.find_template(screenshot, f"assets/Buttons/{go_btn}.png", threshold=0.8)
            if pos:
                self.logger.info(f"{go_btn} button found from goal function")
                cv2.circle(frame, pos, 10, (100, 255, 100), -1)
                self.click_template(pos, go_btn)
                time.sleep(1)
                return
        
        # Default: check stats then go to energy function
        self.logger.info("No race_day/event/infirmary/Go detected - checking stats")
        self.check_current_stats(screenshot, frame)
        self.energy_function(screenshot, frame)
    
    def handle_race_entry(self, screenshot: Image.Image, frame: np.ndarray):
        """Handle Race button clicking after RaceDay, then proceed to race_function."""
        self.logger.info("=== Race Entry - Clicking Race Button ===")
        
        time.sleep(1)
        screenshot, frame = self.capture_and_visualize()
        
        # Multi-method detection: YOLO, Template matching, OCR
        race_pos = None
        detection_method = None
        
        # 1. Try YOLO detection for race_green
        race_detections = self.detect_yolo(screenshot, labels=['race_green'])
        if race_detections.get('race_green') and race_detections['race_green']:
            race_det = race_detections['race_green'][0]
            x1, y1, x2, y2 = race_det['box']
            race_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
            detection_method = "YOLO"
            self.logger.info("Race button detected (YOLO) - clicking first time")
            self.visualize_detections(frame, race_detections['race_green'], (0, 255, 0))
        
        # 2. Try template matching for Race.png
        if not race_pos:
            race_pos = self.find_template(screenshot, "assets/Buttons/Race.png", threshold=0.8)
            if race_pos:
                detection_method = "Template"
                self.logger.info("Race button found (template) - clicking first time")
                cv2.circle(frame, race_pos, 10, (0, 255, 0), -1)
        
        # 3. Try OCR fallback - search for "Race" text
        if not race_pos:
            try:
                img_array = np.array(screenshot)
                ocr_results = reader.readtext(img_array)
                
                for (bbox, text, prob) in ocr_results:
                    # Look for exact "Race" button (not "Race List" or other variants)
                    if text.strip().lower() == "race" and prob > 0.5:
                        # Calculate center of bounding box
                        points = np.array(bbox)
                        cx = int(points[:, 0].mean())
                        cy = int(points[:, 1].mean())
                        
                        # Validate coordinates are within expected region (443, 888 to 672, 955)
                        if 443 <= cx <= 672 and 888 <= cy <= 955:
                            race_pos = (cx, cy)
                            detection_method = "OCR"
                            self.logger.info(f"Race button found via OCR: '{text}' at ({cx}, {cy}) - VALID")
                            cv2.rectangle(frame, tuple(points[0].astype(int)), tuple(points[2].astype(int)), (255, 0, 255), 2)
                            break
                        else:
                            self.logger.warning(f"Race button OCR found '{text}' at ({cx}, {cy}) but outside valid region (443-672, 888-955) - REJECTED")
            except Exception as e:
                self.logger.warning(f"OCR detection failed: {e}")
        
        # 4. Fallback to known coordinates (443, 888 to 672, 955)
        if not race_pos:
            race_pos = ((443 + 672) // 2, (888 + 955) // 2)  # Center of region
            detection_method = "Coordinates"
            self.logger.info(f"Using fallback Race button coordinates: {race_pos}")
            cv2.rectangle(frame, (443, 888), (672, 955), (0, 255, 255), 2)
        
        # Click first time
        self.logger.info(f"Clicking Race button (first time) at {race_pos} via {detection_method}")
        self.automation.click(race_pos[0], race_pos[1])
        time.sleep(0.5)
        
        # 2 second delay
        time.sleep(2)
        screenshot, frame = self.capture_and_visualize()
        
        # Re-detect and click second time using same multi-method approach
        race_pos = None
        
        # 1. YOLO
        race_detections = self.detect_yolo(screenshot, labels=['race_green'])
        if race_detections.get('race_green') and race_detections['race_green']:
            race_det = race_detections['race_green'][0]
            x1, y1, x2, y2 = race_det['box']
            race_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.logger.info("Race button detected (YOLO) - clicking second time")
            self.visualize_detections(frame, race_detections['race_green'], (0, 255, 0))
        
        # 2. Template
        if not race_pos:
            race_pos = self.find_template(screenshot, "assets/Buttons/Race.png", threshold=0.8)
            if race_pos:
                self.logger.info("Race button found (template) - clicking second time")
                cv2.circle(frame, race_pos, 10, (0, 255, 0), -1)
        
        # 3. OCR
        if not race_pos:
            try:
                img_array = np.array(screenshot)
                ocr_results = reader.readtext(img_array)
                for (bbox, text, prob) in ocr_results:
                    # Look for exact "Race" button
                    if text.strip().lower() == "race" and prob > 0.5:
                        points = np.array(bbox)
                        cx = int(points[:, 0].mean())
                        cy = int(points[:, 1].mean())
                        
                        # Validate coordinates are within expected region for SECOND click (568, 738 to 806, 806)
                        if 568 <= cx <= 806 and 738 <= cy <= 806:
                            race_pos = (cx, cy)
                            self.logger.info(f"Race button found via OCR (second click): '{text}' at ({cx}, {cy}) - VALID")
                            break
                        else:
                            self.logger.warning(f"Race button OCR found '{text}' at ({cx}, {cy}) but outside valid region (568-806, 738-806) - REJECTED")
            except Exception as e:
                self.logger.warning(f"OCR detection failed: {e}")
        
        # 4. Fallback to known coordinates for SECOND click (568, 738 to 806, 806)
        if not race_pos:
            race_pos = ((568 + 806) // 2, (738 + 806) // 2)  # Center = (687, 772)
            self.logger.info(f"Using fallback Race button coordinates (second click): {race_pos}")
            cv2.rectangle(frame, (568, 738), (806, 806), (0, 255, 255), 2)
        
        # Click second time if found
        if race_pos:
            self.logger.info(f"Clicking Race button (second time) at {race_pos}")
            self.automation.click(race_pos[0], race_pos[1])
            time.sleep(0.5)
        else:
            self.logger.warning("Race button not found for second click")
        
        # 5 second delay after second click
        time.sleep(5)
        
        # Pass to race function
        self.race_function(screenshot, frame)
    
    def handle_fan_race(self, screenshot: Image.Image, frame: np.ndarray):
        """Handle race selection for fan earning."""
        self.logger.info("=== Handling Fan Race ===")
        
        # Check for race button (Race.png template or race_green YOLO)
        race_pos = self.find_template(screenshot, "assets/Buttons/Race.png", threshold=0.8)
        race_detections = self.detect_yolo(screenshot, labels=['race_green'])
        
        if race_pos or (race_detections.get('race_green') and race_detections['race_green']):
            # Click first time
            if race_pos:
                self.logger.info("Race button found (template) - clicking first time")
                self.click_template(race_pos, "Race")
            else:
                self.logger.info("Race button detected (YOLO) - clicking first time")
                self.click_detection(race_detections['race_green'][0])
            
            # 2 second delay
            time.sleep(2)
            screenshot, frame = self.capture_and_visualize()
            
            # Re-detect and click second time
            race_pos = self.find_template(screenshot, "assets/Buttons/Race.png", threshold=0.8)
            race_detections = self.detect_yolo(screenshot, labels=['race_green'])
            
            if race_pos:
                self.logger.info("Race button found (template) - clicking second time")
                self.click_template(race_pos, "Race")
            elif race_detections.get('race_green') and race_detections['race_green']:
                self.logger.info("Race button detected (YOLO) - clicking second time")
                self.click_detection(race_detections['race_green'][0])
            
            # 5 second delay after second click
            time.sleep(5)
            
            # Pass to race function
            self.race_function(screenshot, frame)
    
    def select_best_fan_race(self, screenshot: Image.Image, frame: np.ndarray):
        """Select race with highest fans and matching terrain/distance stars."""
        self.logger.info("=== Selecting Best Fan Race ===")
        
        # Detect race boxes
        race_box_detections = self.detect_yolo(screenshot, labels=['race_box'])
        if 'race_box' not in race_box_detections:
            self.logger.warning("No race boxes detected")
            return
        
        # Detect fans, terrain_star, distance_star
        fan_detections = self.detect_yolo(screenshot, labels=['fans'])
        terrain_star_detections = self.detect_yolo(screenshot, labels=['terrian_star'])  # Note: typo in yaml
        distance_star_detections = self.detect_yolo(screenshot, labels=['distance_star'])
        
        best_race = None
        best_fans = -1
        
        for race_box in race_box_detections['race_box']:
            rx1, ry1, rx2, ry2 = race_box['box']
            
            # Find fans in this race box
            race_fans = 0
            for fan_det in fan_detections.get('fans', []):
                fx1, fy1, fx2, fy2 = fan_det['box']
                # Check if fan detection is within race box
                if ry1 <= fy1 <= ry2 and ry1 <= fy2 <= ry2:
                    fan_region = screenshot.crop((fx1, fy1, fx2, fy2))
                    race_fans = extract_number(fan_region)
                    self.logger.info(f"Race box at ({rx1},{ry1}) has {race_fans} fans")
                    break
            
            # Check for stars
            has_terrain_star = False
            has_distance_star = False
            
            for star_det in terrain_star_detections.get('terrian_star', []):
                sx1, sy1, sx2, sy2 = star_det['box']
                if ry1 <= sy1 <= ry2:
                    has_terrain_star = True
            
            for star_det in distance_star_detections.get('distance_star', []):
                sx1, sy1, sx2, sy2 = star_det['box']
                if ry1 <= sy1 <= ry2:
                    has_distance_star = True
            
            # Prefer races with both stars and higher fans
            if has_terrain_star and has_distance_star and race_fans > best_fans:
                best_race = race_box
                best_fans = race_fans
        
        if best_race:
            self.logger.info(f"Selected race with {best_fans} fans")
            self.visualize_detections(frame, [best_race], (0, 255, 0), "SELECTED")
            self.click_detection(best_race)
            time.sleep(1)
            self.race_function(screenshot, frame)
        else:
            self.logger.warning("No suitable race found")
    
    def handle_g1_race_schedule(self, screenshot: Image.Image, frame: np.ndarray):
        """Handle G1 race scheduling based on races.json."""
        self.logger.info("=== Handling G1 Race Schedule ===")
        
        if not self.current_month:
            self.logger.warning("Current month unknown - cannot schedule G1 race")
            self.energy_function(screenshot, frame)
            return
        
        # Find closest G1 race matching terrain
        # TODO: Implement race scheduling logic based on races.json
        
        self.energy_function(screenshot, frame)
    
    def race_function(self, screenshot: Image.Image, frame: np.ndarray):
        """Handle race strategy selection (ChangeStrategy) and check race status."""
        self.logger.info("=== On Race Function ===")
        
        time.sleep(1)
        screenshot, frame = self.capture_and_visualize()
        
        # Look for ChangeStrategy button in region (314, 623) to (792, 689)
        pos = self.find_template(screenshot, "assets/Buttons/ChangeStrategy.png", threshold=0.8)
        if pos:
            px, py = pos
            if 314 <= px <= 792 and 623 <= py <= 689:
                self.logger.info("ChangeStrategy button found")
                cv2.circle(frame, pos, 10, (0, 255, 255), -1)
                self.click_template(pos, "ChangeStrategy")
                time.sleep(1)
                
                # Select best style strategy
                screenshot, frame = self.capture_and_visualize()
                self.select_race_strategy(screenshot, frame)
                
                # After selecting strategy, re-capture and continue to check race status
                time.sleep(1)
                screenshot, frame = self.capture_and_visualize()
        
        # Check race status after strategy selection (or if no strategy needed)
        time.sleep(1)
        screenshot, frame = self.capture_and_visualize()
        
        # Check for UViewResults (race already done)
        uview_pos = self.find_template(screenshot, "assets/Buttons/UViewResults.png", threshold=0.8)
        if uview_pos:
            self.logger.info("UViewResults detected - clicking Race to proceed")
            cv2.circle(frame, uview_pos, 10, (255, 100, 0), -1)
            
            # Detect and click Race button
            race_pos = self.find_template(screenshot, "assets/Buttons/Race.png", threshold=0.8)
            race_detections = self.detect_yolo(screenshot, labels=['race_green'])
            
            if race_pos:
                self.click_template(race_pos, "Race")
            elif race_detections.get('race_green') and race_detections['race_green']:
                self.click_detection(race_detections['race_green'][0])
            
            time.sleep(10)
            self.in_race(screenshot, frame)
            return
        
        # Check for AViewResults
        aview_pos = self.find_template(screenshot, "assets/Buttons/AViewResults.png", threshold=0.8)
        if aview_pos:
            self.logger.info("AViewResults detected - clicking it")
            cv2.circle(frame, aview_pos, 10, (255, 100, 0), -1)
            self.click_template(aview_pos, "AViewResults")
            time.sleep(1)
            self.after_race(screenshot, frame)
            return
        
        # No results detected, proceed to in_race
        time.sleep(10)
        self.in_race(screenshot, frame)
    
    def select_race_strategy(self, screenshot: Image.Image, frame: np.ndarray):
        """Select race strategy based on style affinities."""
        self.logger.info("=== Selecting Race Strategy ===")
        
        screenshot, frame = self.capture_and_visualize()
        
        # Get style affinities (from uma_stats)
        style_affinity = self.uma_stats.get('style_affinity', {})
        
        # Find styles with A aptitude, then B
        a_styles = [s for s, v in style_affinity.items() if v == 'A']
        b_styles = [s for s, v in style_affinity.items() if v == 'B']
        
        target_style = None
        if a_styles:
            target_style = a_styles[0]  # Pick first A style
        elif b_styles:
            target_style = b_styles[0]  # Pick first B style
        
        if not target_style:
            self.logger.warning("No suitable style found")
            target_style = 'Front'  # Default
        
        self.logger.info(f"Target style: {target_style}")
        
        # Detect style options (end, late, pace, front)
        style_map = {'Front': 'front', 'Pace': 'pace', 'Late': 'late', 'End': 'end'}
        target_label = style_map.get(target_style, 'front').lower()
        
        style_detections = self.detect_yolo(screenshot, labels=['end', 'late', 'pace', 'front'])
        
        if target_label in style_detections and style_detections[target_label]:
            self.logger.info(f"Found {target_label} style option")
            self.visualize_detections(frame, style_detections[target_label], (255, 0, 255))
            self.click_detection(style_detections[target_label][0])
            time.sleep(0.5)
        
        # Click Confirm button
        pos = self.find_template(screenshot, "assets/Buttons/Confirm.png", threshold=0.8)
        if pos:
            self.logger.info("Confirm button found")
            cv2.circle(frame, pos, 10, (0, 255, 0), -1)
            self.click_template(pos, "Confirm")
            time.sleep(1)
    
    def start_race(self, screenshot: Image.Image, frame: np.ndarray):
        """Start the race."""
        self.logger.info("=== Starting Race ===")
        
        screenshot, frame = self.capture_and_visualize()
        
        # Check for UViewResults (race already done)
        pos = self.find_template(screenshot, "assets/Buttons/UViewResults.png", threshold=0.8)
        if pos:
            self.logger.info("UViewResults detected - race already complete")
            # Detect Race button to proceed
            race_pos = self.find_template(screenshot, "assets/Buttons/Race.png", threshold=0.8)
            if race_pos:
                self.click_template(race_pos, "Race")
                time.sleep(1)
                self.in_race(screenshot, frame)
            return
        
        # Check for AViewResults
        pos = self.find_template(screenshot, "assets/Buttons/AViewResults.png", threshold=0.8)
        if pos:
            self.logger.info("AViewResults detected")
            self.click_template(pos, "AViewResults")
            time.sleep(1)
            self.after_race(screenshot, frame)
            return
        
        # Click Race button
        race_detections = self.detect_yolo(screenshot, labels=['race_green'])
        if 'race_green' in race_detections and race_detections['race_green']:
            self.logger.info("Race button detected (YOLO)")
            self.visualize_detections(frame, race_detections['race_green'], (0, 255, 0))
            self.click_detection(race_detections['race_green'][0])
            time.sleep(1)
            
            # Click again
            screenshot, frame = self.capture_and_visualize()
            race_detections = self.detect_yolo(screenshot, labels=['race_green'])
            if 'race_green' in race_detections and race_detections['race_green']:
                self.click_detection(race_detections['race_green'][0])
                time.sleep(1)
        else:
            # Try template
            pos = self.find_template(screenshot, "assets/Buttons/Race.png", threshold=0.8)
            if pos:
                self.logger.info("Race button found (template)")
                self.click_template(pos, "Race")
                time.sleep(1)
                
                # Click again
                pos = self.find_template(screenshot, "assets/Buttons/Race.png", threshold=0.8)
                if pos:
                    self.click_template(pos, "Race")
                    time.sleep(1)
        
        # Wait 10 seconds for race animations to start
        self.logger.info("Waiting 10 seconds before entering race...")
        time.sleep(10)
        
        self.in_race(screenshot, frame)
    
    def in_race(self, screenshot: Image.Image, frame: np.ndarray):
        """Handle in-race interactions - wait for Race2, then spam Skip until Next button."""
        self.logger.info("=== In Race ===")
        
        # Phase 1: Wait for Race2 button
        self.logger.info("Phase 1: Waiting for Race2 button...")
        race2_detected = False
        max_wait = 120  # Maximum wait time for Race2
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            screenshot, frame = self.capture_and_visualize()
            
            # Check for Race2 button (template matching only)
            pos = self.find_template(screenshot, "assets/Buttons/Race2.png", threshold=0.8)
            if pos:
                self.logger.info("Race2 button detected - clicking it")
                cv2.circle(frame, pos, 10, (0, 255, 255), -1)
                self.click_template(pos, "Race2")
                time.sleep(1)
                race2_detected = True
                break
            
            time.sleep(0.5)
        
        if not race2_detected:
            self.logger.warning("Race2 button not detected within timeout")
        
        # Phase 2: Continuously press Skip until Next button appears
        self.logger.info("Phase 2: Spamming Skip button until Next button appears...")
        next_detected = False
        
        while not next_detected:
            screenshot, frame = self.capture_and_visualize()
            
            # Check for TryAgain and Cancel buttons (race failure scenario)
            try_again_pos = self.find_template(screenshot, "assets/Buttons/TryAgain.png", threshold=0.8)
            cancel_pos = self.find_template(screenshot, "assets/Buttons/Cancel.png", threshold=0.8)
            
            if try_again_pos or cancel_pos:
                self.logger.warning("Race failure detected (TryAgain/Cancel buttons found)")
                
                # Click Cancel button
                if cancel_pos:
                    self.logger.info("Clicking Cancel button")
                    cv2.circle(frame, cancel_pos, 10, (0, 0, 255), -1)
                    self.click_template(cancel_pos, "Cancel")
                    time.sleep(2)
                
                # Click Next button
                screenshot, frame = self.capture_and_visualize()
                next_pos = self.find_template(screenshot, "assets/Buttons/Next.png", threshold=0.8)
                if next_pos:
                    self.logger.info("Clicking Next button after cancel")
                    cv2.circle(frame, next_pos, 10, (0, 255, 0), -1)
                    self.click_template(next_pos, "Next")
                    time.sleep(3)
                else:
                    self.logger.warning("Next button not found after cancel - using fallback coordinates")
                    self.automation.click(553, 994)
                    time.sleep(3)
                
                # Click RaceNext button - try multiple methods
                screenshot, frame = self.capture_and_visualize()
                racenext_clicked = False
                
                # Try template matching
                racenext_pos = self.find_template(screenshot, "assets/Buttons/RaceNext.png", threshold=0.8)
                if racenext_pos:
                    self.logger.info("Clicking RaceNext button after cancel (template)")
                    cv2.circle(frame, racenext_pos, 10, (0, 255, 255), -1)
                    self.click_template(racenext_pos, "RaceNext")
                    racenext_clicked = True
                    time.sleep(3)
                
                # Try coordinate fallback if template didn't find it
                if not racenext_clicked:
                    self.logger.info("RaceNext not found via template - using fallback coordinates (676, 997)")
                    cv2.rectangle(frame, (564, 944), (789, 1050), (0, 255, 255), 2)
                    self.automation.click(676, 997)
                    time.sleep(3)
                
                # Click another Next button - try multiple methods
                screenshot, frame = self.capture_and_visualize()
                next_clicked = False
                
                # Try template matching
                next_pos = self.find_template(screenshot, "assets/Buttons/Next.png", threshold=0.8)
                if next_pos:
                    self.logger.info("Clicking final Next button after cancel (template)")
                    cv2.circle(frame, next_pos, 10, (0, 255, 0), -1)
                    self.click_template(next_pos, "Next")
                    next_clicked = True
                    time.sleep(3)
                
                # Try coordinate fallback if template didn't find it
                if not next_clicked:
                    self.logger.info("Final Next not found via template - using fallback coordinates (553, 994)")
                    self.automation.click(553, 994)
                    time.sleep(3)
                
                # End the program
                self.logger.info("Race failure handled - ending program")
                self.stop_requested = True
                return
            
            # Check for Next button (template matching)
            next_pos = self.find_template(screenshot, "assets/Buttons/Next.png", threshold=0.8)
            if next_pos:
                self.logger.info("Next button detected - exiting race loop")
                cv2.circle(frame, next_pos, 10, (0, 255, 0), -1)
                next_detected = True
                break
            
            # Press Skip button - try multiple methods
            skip_pressed = False
            
            # 1. Try Skip.png template
            skip_pos = self.find_template(screenshot, "assets/Buttons/Skip.png", threshold=0.8)
            if skip_pos:
                self.logger.info("Skip button found (template) - pressing it")
                cv2.circle(frame, skip_pos, 10, (255, 255, 0), -1)
                self.click_template(skip_pos, "Skip")
                skip_pressed = True
            
            # 2. Try SkipBig.png template
            if not skip_pressed:
                skipbig_pos = self.find_template(screenshot, "assets/Buttons/SkipBig.png", threshold=0.8)
                if skipbig_pos:
                    self.logger.info("SkipBig button found (template) - pressing it")
                    cv2.circle(frame, skipbig_pos, 10, (255, 255, 0), -1)
                    self.click_template(skipbig_pos, "SkipBig")
                    skip_pressed = True
            
            # 3. Fallback to known coordinates (1624, 916 to 1745, 1041)
            if not skip_pressed:
                skip_pos = ((1624 + 1745) // 2, (916 + 1041) // 2)  # Center = (1684, 978)
                self.logger.info(f"Using fallback Skip button coordinates: {skip_pos}")
                cv2.rectangle(frame, (1624, 916), (1745, 1041), (0, 255, 255), 2)
                self.automation.click(skip_pos[0], skip_pos[1])
                time.sleep(0.5)
            
            # Wait 1 second before next Skip attempt
            time.sleep(1)
        
        self.logger.info("Race complete - proceeding to after_race")
        self.after_race(screenshot, frame)
    
    def after_race(self, screenshot: Image.Image, frame: np.ndarray):
        """Handle post-race interactions."""
        self.logger.info("=== After Race ===")
        
        racenext_clicked = False  # Track if RaceNext was already clicked
        max_clicks = 20
        
        for i in range(max_clicks):
            screenshot, frame = self.capture_and_visualize()
            
            # Check for Next button
            next_detections = self.detect_yolo(screenshot, labels=['Next'])
            if 'Next' in next_detections and next_detections['Next']:
                self.logger.info("Next button detected (YOLO)")
                self.visualize_detections(frame, next_detections['Next'], (100, 255, 100))
                self.click_detection(next_detections['Next'][0])
                time.sleep(0.5)
                continue
            
            pos = self.find_template(screenshot, "assets/Buttons/Next.png", threshold=0.8)
            if pos:
                self.logger.info("Next button found (template)")
                cv2.circle(frame, pos, 10, (100, 255, 100), -1)
                self.click_template(pos, "Next")
                time.sleep(0.5)
                continue
            
            # Check for RaceNext button (only if not already clicked)
            if not racenext_clicked:
                racenext_pos = self.find_template(screenshot, "assets/Buttons/RaceNext.png", threshold=0.8)
                if racenext_pos:
                    self.logger.info("RaceNext button found (template) - clicking and waiting 4s")
                    cv2.circle(frame, racenext_pos, 10, (100, 255, 100), -1)
                    self.click_template(racenext_pos, "RaceNext")
                    racenext_clicked = True
                    time.sleep(4)  # Wait 4 seconds for potential Next buttons to appear
                    continue  # Continue loop to check for additional Next buttons
                elif i > 5:  # After a few tries, use fallback
                    # Fallback to coordinates (564, 944 to 789, 1050)
                    racenext_coords = ((564 + 789) // 2, (944 + 1050) // 2)  # Center = (676, 997)
                    self.logger.info(f"Using fallback RaceNext coordinates: {racenext_coords} - waiting 4s")
                    cv2.rectangle(frame, (564, 944), (789, 1050), (100, 255, 255), 2)
                    self.automation.click(racenext_coords[0], racenext_coords[1])
                    racenext_clicked = True
                    time.sleep(4)  # Wait 4 seconds for potential Next buttons to appear
                    continue  # Continue loop to check for additional Next buttons
            
            # Check for Next_race (indicates we're done)
            next_race_det = self.detect_yolo(screenshot, labels=['Next_race'])
            if 'Next_race' in next_race_det and next_race_det['Next_race']:
                self.logger.info("Next_race detected - race complete")
                self.visualize_detections(frame, next_race_det['Next_race'], (0, 255, 0))
                self.click_detection(next_race_det['Next_race'][0])
                time.sleep(1)
                break
            
            time.sleep(0.5)
    
    def energy_function(self, screenshot: Image.Image, frame: np.ndarray):
        """Check energy and decide to rest or train."""
        self.logger.info("=== Energy Function ===")
        
        # Read energy using EnergyReader from screen_capture.py
        from screen_capture import EnergyReader
        energy_reader = EnergyReader(monitor_num=self.monitor)
        energy_info = energy_reader.read()
        current_energy = energy_info.get('energy', 60)
        
        self.logger.info(f"Current energy: {current_energy} (filled: {energy_info.get('filled_pixels')}/{energy_info.get('total_pixels')})")
        
        # Also visualize energy bar if YOLO detects it
        energy_detections = self.detect_yolo(screenshot, labels=['Energy'])
        if 'Energy' in energy_detections and energy_detections['Energy']:
            self.visualize_detections(frame, energy_detections['Energy'], (255, 255, 0))
        
        # Check for low failure rate training (<10%)
        low_fail_training = self.check_low_failure_training(screenshot, frame)
        
        if current_energy < 50:
            if low_fail_training:
                self.logger.info("Energy low but found <10% failure training")
                self.training_function(screenshot, frame)
            else:
                self.logger.info("Energy low - going to rest")
                self.use_rest(screenshot, frame)
        else:
            self.logger.info("Energy sufficient - going to training")
            self.training_function(screenshot, frame)
    
    def check_low_failure_training(self, screenshot: Image.Image, frame: np.ndarray) -> bool:
        """Check if there's a training option with <10% failure rate."""
        # Detect fail rate indicators
        fail_detections = self.detect_yolo(screenshot, labels=['fail'])
        if 'fail' in fail_detections:
            for fail_det in fail_detections['fail']:
                fx1, fy1, fx2, fy2 = fail_det['box']
                fail_region = screenshot.crop((fx1, fy1, fx2, fy2))
                fail_rate = extract_number(fail_region)
                self.logger.info(f"Found failure rate: {fail_rate}%")
                if 0 <= fail_rate < 10:
                    return True
        return False
    
    def use_rest(self, screenshot: Image.Image, frame: np.ndarray):
        """Click rest button to recover energy."""
        self.logger.info("=== Use Rest ===")
        
        rest_pos = None
        
        # Try YOLO
        rest_detections = self.detect_yolo(screenshot, labels=['rest'])
        if 'rest' in rest_detections and rest_detections['rest']:
            self.logger.info("Rest detected (YOLO)")
            self.visualize_detections(frame, rest_detections['rest'], (0, 255, 255))
            self.click_detection(rest_detections['rest'][0])
            time.sleep(3)
            return
        
        # Try template
        rest_pos = self.find_template(screenshot, "assets/Buttons/Rest.png", threshold=0.8)
        if rest_pos:
            self.logger.info("Rest found (template)")
            cv2.circle(frame, rest_pos, 10, (0, 255, 255), -1)
            self.click_template(rest_pos, "Rest")
            time.sleep(3)
            return
        
        # Try OCR with coordinate validation
        try:
            img_array = np.array(screenshot)
            ocr_results = reader.readtext(img_array)
            for (bbox, text, prob) in ocr_results:
                if text.strip().lower() == "rest" and prob > 0.5:
                    points = np.array(bbox)
                    cx = int(points[:, 0].mean())
                    cy = int(points[:, 1].mean())
                    
                    # Validate coordinates are within expected region (278, 802 to 420, 880)
                    if 278 <= cx <= 420 and 802 <= cy <= 880:
                        rest_pos = (cx, cy)
                        self.logger.info(f"Rest button found via OCR: '{text}' at ({cx}, {cy}) - VALID")
                        cv2.rectangle(frame, tuple(points[0].astype(int)), tuple(points[2].astype(int)), (255, 0, 255), 2)
                        break
                    else:
                        self.logger.warning(f"Rest button OCR found '{text}' at ({cx}, {cy}) but outside valid region (278-420, 802-880) - REJECTED")
        except Exception as e:
            self.logger.warning(f"OCR detection failed: {e}")
        
        # Fallback to known coordinates (278, 802 to 420, 880)
        if not rest_pos:
            rest_pos = ((278 + 420) // 2, (802 + 880) // 2)  # Center = (349, 841)
            self.logger.info(f"Using fallback Rest button coordinates: {rest_pos}")
            cv2.rectangle(frame, (278, 802), (420, 880), (0, 255, 255), 2)
        
        # Click rest button
        self.logger.info(f"Clicking Rest button at {rest_pos}")
        self.automation.click(rest_pos[0], rest_pos[1])
        time.sleep(3)
    
    def training_function(self, screenshot: Image.Image, frame: np.ndarray):
        """Handle training button detection and click."""
        self.logger.info("=== Training Function ===")
        
        training_clicked = False
        
        # Method 1: Detect training button via YOLO
        training_detections = self.detect_yolo(screenshot, labels=['training', 'Training'])
        
        if training_detections:
            for label, dets in training_detections.items():
                if dets:
                    self.logger.info(f"Training detected ({label}) via YOLO at {dets[0]['box']}")
                    self.visualize_detections(frame, dets, (255, 0, 255))
                    self.click_detection(dets[0])
                    training_clicked = True
                    time.sleep(1.5)
                    break
        
        # Method 2: Template matching
        if not training_clicked:
            pos = self.find_template(screenshot, "assets/Buttons/Training.png", threshold=0.8)
            if pos:
                self.logger.info(f"Training found via template at {pos}")
                cv2.circle(frame, pos, 10, (255, 0, 255), -1)
                self.click_template(pos, "Training")
                training_clicked = True
                time.sleep(1.5)
        
        # Method 3: OCR detection of "training" text + coordinate fallback
        if not training_clicked:
            # Try OCR in the button region
            button_region = screenshot.crop((439, 784, 660, 876))
            button_text = extract_text(button_region).lower()
            
            if "training" in button_text or not training_clicked:
                # Click center of training button region
                center_x = (439 + 660) // 2
                center_y = (784 + 876) // 2
                self.logger.info(f"Training button detected via OCR/fallback - clicking ({center_x}, {center_y})")
                cv2.circle(frame, (center_x, center_y), 15, (255, 0, 255), 3)
                cv2.putText(frame, "Training", (center_x - 40, center_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                self.automation.click(center_x, center_y)
                training_clicked = True
                time.sleep(1.5)
        
        # Pass to in_training to select stat
        if training_clicked:
            self.in_training(screenshot, frame)
    
    def in_training(self, screenshot: Image.Image, frame: np.ndarray):
        """Inside training screen - analyze options and select best."""
        self.logger.info("=== In Training ===")
        
        screenshot, frame = self.capture_and_visualize()
        
        # Detect all 5 stat positions (speed, stamina, power, guts, wit)
        stat_labels = ['speed', 'stamina', 'power', 'guts', 'wit']
        stat_detections = self.detect_yolo(screenshot, labels=stat_labels)
        
        # Log what was detected
        detected_stats = [stat for stat in stat_labels if stat in stat_detections and stat_detections[stat]]
        self.logger.info(f"Detected training stat buttons: {detected_stats}")
        
        # Prepare positions: use YOLO if available, otherwise use fallback coordinates
        stat_positions = {}
        fallback_coords = {
            'speed': (350, 870),
            'stamina': (450, 870),
            'power': (550, 870),
            'guts': (650, 870),
            'wit': (750, 870)
        }
        
        # Expected X ranges for validation (±50 pixels from fallback)
        expected_x_ranges = {
            'speed': (300, 400),
            'stamina': (400, 500),
            'power': (500, 600),
            'guts': (600, 700),
            'wit': (700, 800)
        }
        
        for stat_name in stat_labels:
            if stat_name in stat_detections and stat_detections[stat_name]:
                # Use YOLO detection
                stat_det = stat_detections[stat_name][0]
                x1, y1, x2, y2 = stat_det['box']
                stat_center_x = (x1 + x2) // 2
                stat_center_y = (y1 + y2) // 2
                
                # Validate X position is within expected range
                min_x, max_x = expected_x_ranges[stat_name]
                if min_x <= stat_center_x <= max_x:
                    stat_positions[stat_name] = {
                        'position': (stat_center_x, stat_center_y),
                        'detection': stat_det,
                        'method': 'YOLO'
                    }
                    self.logger.debug(f"{stat_name}: YOLO detected at ({stat_center_x}, {stat_center_y}) - VALID")
                else:
                    # YOLO detection outside expected range - use fallback
                    fallback_pos = fallback_coords[stat_name]
                    stat_positions[stat_name] = {
                        'position': fallback_pos,
                        'detection': None,
                        'method': 'fallback'
                    }
                    self.logger.warning(f"{stat_name}: YOLO detected at ({stat_center_x}, {stat_center_y}) outside range ({min_x}-{max_x}) - using fallback {fallback_pos}")
            else:
                # Use fallback coordinates
                fallback_pos = fallback_coords[stat_name]
                stat_positions[stat_name] = {
                    'position': fallback_pos,
                    'detection': None,
                    'method': 'fallback'
                }
                self.logger.debug(f"{stat_name}: Using fallback coords {fallback_pos}")
        
        if not stat_positions:
            self.logger.warning("No training stat buttons detected and no fallback positions available")
            return
        
        training_options = []
        
        # Hold left click and move through each stat sequentially
        import pyautogui
        
        for stat_name in stat_labels:
            if stat_name not in stat_positions:
                continue
            
            stat_info = stat_positions[stat_name]
            stat_center_x, stat_center_y = stat_info['position']
            
            self.logger.info(f"Checking {stat_name} training at ({stat_center_x}, {stat_center_y}) [{stat_info['method']}]")
            
            # Hold left click at this position (or move to it if already holding)
            is_first_stat = len(training_options) == 0
            
            if is_first_stat:
                # First stat - start holding
                pyautogui.mouseDown(stat_center_x, stat_center_y)
                self.logger.debug(f"Started holding at {stat_name}")
            else:
                # Subsequent stats - move while holding
                pyautogui.moveTo(stat_center_x, stat_center_y, duration=0.3)
                self.logger.debug(f"Moved to {stat_name} while holding")
            
            time.sleep(0.5)
            
            # Read data with retry logic
            max_retries = 2
            failure_rate = None
            stat_additions = None
            
            for attempt in range(max_retries):
                hover_screenshot, hover_frame = self.capture_and_visualize()
                
                # Read failure rate from region (273, 767, 841, 815)
                # Try YOLO first
                fail_detections = self.detect_yolo(hover_screenshot, labels=['fail'])
                if 'fail' in fail_detections and fail_detections['fail']:
                    fx1, fy1, fx2, fy2 = fail_detections['fail'][0]['box']
                    fail_region = hover_screenshot.crop((fx1, fy1, fx2, fy2))
                    from ocr import extract_percentage
                    failure_rate = extract_percentage(fail_region)
                    if failure_rate >= 0:
                        self.logger.debug(f"Failure rate for {stat_name} (YOLO): {failure_rate}%")
                    else:
                        failure_rate = None
                
                if failure_rate is None:
                    # Fallback: Use fixed region (273, 767, 841, 815)
                    fail_region = hover_screenshot.crop((273, 767, 841, 815))
                    fail_region.save(f"debug/fail_rate_{stat_name}.png")
                    from ocr import extract_percentage
                    failure_rate = extract_percentage(fail_region)
                    if failure_rate >= 0:
                        self.logger.debug(f"Failure rate for {stat_name} (coords): {failure_rate}%")
                    else:
                        failure_rate = None
                
                # Read stat additions from region 270,647 to 745,695 using improved method
                from read_stat_additions import read_stat_additions
                try:
                    stat_additions = read_stat_additions(hover_screenshot, 270, 647, 745, 695)
                    self.logger.debug(f"Stat additions for {stat_name}: {stat_additions}")
                except Exception as e:
                    self.logger.error(f"Error reading stat additions: {e}")
                    stat_additions = None
                
                # Check if we got both pieces of data
                if failure_rate is not None and stat_additions is not None:
                    break
                
                if attempt < max_retries - 1:
                    self.logger.warning(f"Incomplete data for {stat_name} (attempt {attempt+1}/{max_retries}), retrying...")
                    time.sleep(0.3)
            
            # Store option if we have valid data (use defaults if necessary)
            if failure_rate is None:
                self.logger.warning(f"Could not read failure rate for {stat_name} - using default 20%")
                failure_rate = 20
            
            if stat_additions is not None:
                self.logger.info(f"{stat_name}: Fail={failure_rate}%, Stats={stat_additions}")
                training_options.append({
                    'stat_name': stat_name,
                    'position': (stat_center_x, stat_center_y),
                    'failure_rate': failure_rate,
                    'stat_additions': stat_additions,
                    'detection': stat_info['detection']
                })
            else:
                self.logger.warning(f"Could not get stat additions for {stat_name} - skipping")
        
        # Release mouse after checking all stats
        pyautogui.mouseUp()
        self.logger.debug("Released mouse after checking all stats")
        time.sleep(0.3)
        
        # Select best training based on stat_treshold priorities
        best_training = self.select_best_training(training_options)
        
        if best_training:
            self.logger.info(f"Selected training: {best_training['stat_name']}")
            px, py = best_training['position']
            
            # Hover over best choice to show info (changes view)
            self.automation.move_to(px, py, duration=0.3)
            time.sleep(0.5)
            
            # First click changes the view to this training option
            self.automation.click(px, py)
            self.logger.info(f"First click on {best_training['stat_name']} at ({px}, {py}) - changed view")
            time.sleep(0.3)
            
            # Second click confirms the selection
            self.automation.click(px, py)
            self.logger.info(f"Second click on {best_training['stat_name']} at ({px}, {py}) - confirmed selection")
            time.sleep(5)  # Wait for training animation to complete
            
            # Return to menu (automation loop continues)
        else:
            self.logger.warning("No suitable training found")
    
    def select_best_training(self, training_options: List[Dict]) -> Optional[Dict]:
        """Select best training based on current needs."""
        if not training_options:
            return None
        
        # Filter by failure rate < 30%
        safe_options = [t for t in training_options if t['failure_rate'] < 30]
        if not safe_options:
            safe_options = training_options  # Use all if none are safe
        
        # TODO: Implement sophisticated selection based on stat_treshold
        # For now, select the one with lowest failure rate and highest total stat gain
        
        best = None
        best_score = -1
        
        for option in safe_options:
            # Calculate total stat gain
            total_gain = sum(option['stat_additions'].values())
            # Score: prefer lower failure and higher gain
            score = total_gain - (option['failure_rate'] / 10)
            
            if score > best_score:
                best_score = score
                best = option
        
        return best
    
    def check_for_event(self, screenshot: Image.Image, frame: np.ndarray) -> bool:
        """Check if there's an event on screen."""
        # If we've detected events 3+ times consecutively, ignore and reset counter
        if self.consecutive_event_detections >= 3:
            self.logger.warning(f"Detected events {self.consecutive_event_detections} times consecutively - ignoring event and resetting counter")
            self.consecutive_event_detections = 0
            return False
        
        # Method 1: Look for EventChoice icon in full screen, but validate position
        icon_path = "assets/Icons/EventChoice.png"
        
        if Path(icon_path).exists():
            pos = self.find_template(screenshot, icon_path, threshold=0.6)
            if pos:
                # Validate icon position - EventChoice icons should be on the left side of screen (X < 800)
                if pos[0] > 800:
                    self.logger.debug(f"EventChoice icon found at {pos} but X > 800 - likely false positive, ignoring")
                    return False
                
                # Verify event text is readable before returning True
                text_region = screenshot.crop((257, 205, 512, 276))
                text_region_np = np.array(text_region)
                
                try:
                    results = reader.readtext(text_region_np)
                    for (bbox, text, confidence) in results:
                        if text.strip():  # Any non-empty text
                            self.logger.info(f"Event choice icon detected at {pos} with valid text: '{text}'")
                            cv2.circle(frame, pos, 10, (255, 165, 0), -1)  # Orange circle
                            return True
                except Exception as e:
                    self.logger.debug(f"OCR event text verification error: {e}")
                
                self.logger.debug(f"EventChoice icon found at {pos} but no valid event text - ignoring")
                return False
        
        # Method 2: OCR text detection in region (257, 205 to 512, 276) looking for "event"
        text_region = screenshot.crop((257, 205, 512, 276))
        text_region_np = np.array(text_region)
        
        try:
            results = reader.readtext(text_region_np)
            for (bbox, text, confidence) in results:
                if "event" in text.strip().lower():
                    self.logger.info(f"Event text detected via OCR: '{text}' (confidence: {confidence:.2f})")
                    # Draw detection on frame (adjust coordinates)
                    adjusted_x = int(bbox[0][0]) + 257
                    adjusted_y = int(bbox[0][1]) + 205
                    cv2.circle(frame, (adjusted_x, adjusted_y), 10, (255, 165, 0), -1)
                    return True
        except Exception as e:
            self.logger.debug(f"OCR event detection error: {e}")
        
        # If no event found
        return False
    
    def handle_event(self, screenshot: Image.Image, frame: np.ndarray):
        """Handle event choice selection."""
        self.logger.info("=== Handling Event ===")
        
        # Read event text from region (257, 205 to 512, 276)
        event_region = screenshot.crop((257, 205, 512, 276))
        event_text = extract_text(event_region)
        self.logger.info(f"Event text: {event_text}")
        
        # If event text is empty, skip event handling
        if not event_text or not event_text.strip():
            self.logger.info("Event text is empty - skipping event handling")
            return
        
        # Save debug image
        event_region.save("debug/event_text_region.png")
        
        # Find matching event in events.json
        best_choice = self.find_best_event_choice(event_text)
        
        if best_choice:
            choice_number = int(best_choice.get('choice_number', '1'))
            choice_text = best_choice.get('choice_text', '')
            outcomes = best_choice.get('all_outcomes', '')
            self.logger.info(f"Best choice: #{choice_number} - {choice_text}")
            self.logger.info(f"Expected outcomes: {outcomes}")
            
            # Find EventChoice icons
            icon_path = "assets/Icons/EventChoice.png"
            if Path(icon_path).exists():
                # Find all EventChoice icons (should be multiple)
                screen_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                template = cv2.imread(icon_path)
                
                if template is not None:
                    result = cv2.matchTemplate(screen_cv, template, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.6
                    locations = np.where(result >= threshold)
                    
                    # Group matches by y-coordinate to find distinct choices
                    matches = []
                    for pt in zip(*locations[::-1]):
                        x, y = pt[0] + template.shape[1]//2, pt[1] + template.shape[0]//2
                        # Check if this is a new match (not too close to existing ones)
                        is_new = True
                        for mx, my in matches:
                            if abs(y - my) < 50:  # Within 50px, consider same choice
                                is_new = False
                                break
                        if is_new:
                            matches.append((x, y))
                    
                    # Sort by y-coordinate (top to bottom)
                    matches.sort(key=lambda p: p[1])
                    self.logger.info(f"Found {len(matches)} choice icons at: {matches}")
                    
                    # Check if we have any matches at all
                    if not matches:
                        self.logger.warning("No EventChoice icons detected - clicking center to dismiss")
                        center_x = (239 + 614) // 2
                        center_y = (198 + 244) // 2 + 100
                        self.automation.click(center_x, center_y)
                        time.sleep(1)
                        return
                    
                    # Select the choice based on choice_number (1-indexed)
                    if len(matches) >= choice_number:
                        click_x, click_y = matches[choice_number - 1]
                        self.logger.info(f"Clicking choice {choice_number} at ({click_x}, {click_y})")
                        cv2.circle(frame, (click_x, click_y), 15, (0, 255, 0), 3)
                        cv2.putText(frame, f"Choice {choice_number}", (click_x - 40, click_y - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        self.automation.click(click_x, click_y)
                        time.sleep(1)
                        return  # Successfully clicked
                    else:
                        # Click first choice as fallback
                        self.logger.warning(f"Only {len(matches)} choices found, wanted #{choice_number} - clicking first")
                        click_x, click_y = matches[0]
                        self.logger.info(f"Clicking first choice as fallback at ({click_x}, {click_y})")
                        self.automation.click(click_x, click_y)
                        time.sleep(1)
                        return  # Successfully clicked
                else:
                    self.logger.error(f"Could not load template: {icon_path}")
            else:
                self.logger.warning(f"EventChoice template not found: {icon_path}")
        
        # Fallback: click first EventChoice icon found
        self.logger.warning("No matching event found in database - trying to find any choice icon")
        icon_path = "assets/Icons/EventChoice.png"
        pos = self.find_template(screenshot, icon_path, threshold=0.6)
        if pos:
            self.logger.info(f"Clicking first choice at {pos}")
            cv2.circle(frame, pos, 10, (255, 0, 0), -1)
            self.click_template(pos, "EventChoice")
            time.sleep(1)
        else:
            # Ultimate fallback - click center area to try to dismiss
            self.logger.warning("No icons found - clicking center to try to progress")
            center_x = screenshot.width // 2
            center_y = screenshot.height // 2 + 100
            self.automation.click(center_x, center_y)
            time.sleep(1)
    
    def find_best_event_choice(self, event_text: str) -> Optional[Dict]:
        """Find best event choice from events.json."""
        # Search for matching event in database
        if 'choiceArraySchema' not in self.events_data:
            self.logger.warning("events.json missing choiceArraySchema")
            return None
        
        choices = self.events_data['choiceArraySchema'].get('choices', [])
        
        # Find events with matching text
        matching_choices = []
        event_text_clean = event_text.lower().strip()
        
        for choice in choices:
            event_name = choice.get('event_name', '').lower()
            # Match by event name
            if event_name and (event_name in event_text_clean or event_text_clean in event_name):
                matching_choices.append(choice)
        
        if not matching_choices:
            self.logger.info(f"No exact match found for event: {event_text[:50]}...")
            return None
        
        self.logger.info(f"Found {len(matching_choices)} matching event choices")
        
        # Group choices by event
        events_grouped = {}
        for choice in matching_choices:
            event_name = choice.get('event_name', '')
            if event_name not in events_grouped:
                events_grouped[event_name] = []
            events_grouped[event_name].append(choice)
        
        # Select best choice from the first matching event
        first_event = list(events_grouped.keys())[0]
        event_choices = events_grouped[first_event]
        
        self.logger.info(f"Evaluating {len(event_choices)} choices for event: {first_event}")
        
        # Score each choice
        best_choice = None
        best_score = -999999
        
        # Get current energy (default 60 if unknown)
        current_energy = 60
        
        for choice in event_choices:
            outcomes = choice.get('all_outcomes', '')
            score = 0
            
            # Parse outcomes
            outcome_parts = outcomes.split(';')
            has_negative = False
            energy_gain = 0
            stat_gains = {}
            
            for part in outcome_parts:
                part = part.strip()
                if not part:
                    continue
                
                # Check for negative outcomes
                if part.startswith('-'):
                    has_negative = True
                    self.logger.debug(f"Choice {choice.get('choice_number')} has negative: {part}")
                    score -= 50  # Penalize negative outcomes
                    continue
                
                # Parse stat gains
                if 'Speed' in part:
                    try:
                        val = int(''.join(filter(str.isdigit, part)))
                        stat_gains['speed'] = val
                        score += val * 2  # Prioritize speed
                    except:
                        pass
                elif 'Stamina' in part:
                    try:
                        val = int(''.join(filter(str.isdigit, part)))
                        stat_gains['stamina'] = val
                        score += val * 1.5
                    except:
                        pass
                elif 'Power' in part:
                    try:
                        val = int(''.join(filter(str.isdigit, part)))
                        stat_gains['power'] = val
                        score += val * 1.8
                    except:
                        pass
                elif 'Guts' in part:
                    try:
                        val = int(''.join(filter(str.isdigit, part)))
                        stat_gains['guts'] = val
                        score += val * 1.3
                    except:
                        pass
                elif 'Wit' in part or 'Wits' in part:
                    try:
                        val = int(''.join(filter(str.isdigit, part)))
                        stat_gains['wit'] = val
                        score += val * 1.5
                    except:
                        pass
                elif 'energy' in part.lower():
                    try:
                        val = int(''.join(filter(str.isdigit, part)))
                        energy_gain = val
                        
                        # Check for energy overflow
                        if current_energy + val > 100:
                            overflow = (current_energy + val) - 100
                            score -= overflow * 2  # Penalize overflow
                            self.logger.debug(f"Choice {choice.get('choice_number')} would overflow energy by {overflow}")
                        else:
                            score += val * 0.5  # Energy is less valuable than stats
                    except:
                        pass
            
            # Avoid choices with only energy when we have stats options
            if energy_gain > 0 and not stat_gains:
                score -= 10  # Slight penalty for energy-only choices
            
            self.logger.debug(f"Choice {choice.get('choice_number')}: score={score}, outcomes={outcomes}")
            
            if score > best_score:
                best_score = score
                best_choice = choice
        
        if best_choice:
            self.logger.info(f"Selected choice {best_choice.get('choice_number')} with score {best_score}")
        
        return best_choice
    
    def cleanup(self):
        """Cleanup and save video."""
        self.logger.info("Cleaning up...")
        
        # Unhook keyboard listener
        try:
            keyboard.unhook_all()
        except:
            pass
        
        if self.video_recorder:
            self.video_recorder.release()
            self.logger.info("Video saved")
        
        self.logger.info("Automation finished")


def main():
    """Entry point for game automation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Uma Musume Game Automation")
    parser.add_argument('--model', type=str, default='yolo_project/weights/best.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor number to capture from')
    parser.add_argument('--no-video', action='store_true',
                       help='Disable video recording')
    
    args = parser.parse_args()
    
    automation = GameAutomation(
        model_path=args.model,
        monitor=args.monitor,
        record_video=not args.no_video
    )
    
    automation.run()


if __name__ == "__main__":
    main()
