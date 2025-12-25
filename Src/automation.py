"""Automation system for controlling mouse and keyboard (PyAutoGUI wrapper)."""
import pyautogui
import time
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image


class AutomationController:
    """Wrapper around PyAutoGUI for game/application automation."""
    
    def __init__(self):
        """Initialize automation controller."""
        # Safety settings
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.1  # Pause between actions
        
    def set_pause(self, seconds: float):
        """Set pause duration between actions."""
        pyautogui.PAUSE = seconds
    
    # Mouse Control
    def move_to(self, x: int, y: int, duration: float = 0.5):
        """
        Move mouse to position.
        
        Args:
            x, y: Screen coordinates
            duration: Time to move (seconds)
        """
        pyautogui.moveTo(x, y, duration=duration)
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, 
              button: str = 'left', clicks: int = 1, interval: float = 0.0):
        """
        Click at position.
        
        Args:
            x, y: Position (None = current position)
            button: 'left', 'right', or 'middle'
            clicks: Number of clicks
            interval: Interval between clicks
        """
        if x is not None and y is not None:
            pyautogui.click(x, y, clicks=clicks, interval=interval, button=button)
        else:
            pyautogui.click(clicks=clicks, interval=interval, button=button)
    
    def double_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """Double click at position."""
        self.click(x, y, clicks=2, interval=0.1)
    
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """Right click at position."""
        self.click(x, y, button='right')
    
    def drag_to(self, x: int, y: int, duration: float = 0.5, button: str = 'left'):
        """
        Drag mouse to position.
        
        Args:
            x, y: Target position
            duration: Drag duration
            button: Mouse button to hold
        """
        pyautogui.dragTo(x, y, duration=duration, button=button)
    
    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None):
        """
        Scroll at position.
        
        Args:
            clicks: Scroll amount (positive = up, negative = down)
            x, y: Position to scroll at
        """
        if x is not None and y is not None:
            pyautogui.scroll(clicks, x=x, y=y)
        else:
            pyautogui.scroll(clicks)
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        return pyautogui.position()
    
    # Keyboard Control
    def press_key(self, key: str):
        """
        Press a single key.
        
        Args:
            key: Key name (e.g., 'enter', 'space', 'a', 'ctrl')
        """
        pyautogui.press(key)
    
    def press_keys(self, keys: List[str]):
        """Press multiple keys in sequence."""
        for key in keys:
            pyautogui.press(key)
    
    def hotkey(self, *keys):
        """
        Press hotkey combination.
        
        Args:
            *keys: Keys to press together (e.g., 'ctrl', 'c')
        """
        pyautogui.hotkey(*keys)
    
    def type_text(self, text: str, interval: float = 0.0):
        """
        Type text.
        
        Args:
            text: Text to type
            interval: Interval between keypresses
        """
        pyautogui.write(text, interval=interval)
    
    def key_down(self, key: str):
        """Hold key down."""
        pyautogui.keyDown(key)
    
    def key_up(self, key: str):
        """Release key."""
        pyautogui.keyUp(key)
    
    # Screen and Image Functions
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size."""
        return pyautogui.size()
    
    def screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """
        Take screenshot.
        
        Args:
            region: (left, top, width, height) or None for full screen
            
        Returns:
            PIL Image
        """
        return pyautogui.screenshot(region=region)
    
    def locate_on_screen(
        self,
        template: str,
        confidence: float = 0.9,
        grayscale: bool = True
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate template on screen.
        
        Args:
            template: Path to template image
            confidence: Matching confidence (0-1)
            grayscale: Use grayscale matching
            
        Returns:
            (left, top, width, height) or None
        """
        try:
            result = pyautogui.locateOnScreen(
                template,
                confidence=confidence,
                grayscale=grayscale
            )
            if result:
                return (result.left, result.top, result.width, result.height)
        except Exception as e:
            print(f"Error locating template: {e}")
        return None
    
    def locate_center_on_screen(
        self,
        template: str,
        confidence: float = 0.9
    ) -> Optional[Tuple[int, int]]:
        """
        Locate center of template on screen.
        
        Returns:
            (x, y) or None
        """
        try:
            result = pyautogui.locateCenterOnScreen(
                template,
                confidence=confidence
            )
            if result:
                return (result.x, result.y)
        except Exception as e:
            print(f"Error locating template: {e}")
        return None
    
    def click_on_template(
        self,
        template: str,
        confidence: float = 0.9,
        button: str = 'left'
    ) -> bool:
        """
        Click on template if found.
        
        Returns:
            True if clicked, False if not found
        """
        center = self.locate_center_on_screen(template, confidence)
        if center:
            self.click(center[0], center[1], button=button)
            return True
        return False
    
    # Utility Functions
    def wait(self, seconds: float):
        """Wait for specified time."""
        time.sleep(seconds)
    
    def is_point_on_screen(self, x: int, y: int) -> bool:
        """Check if point is on screen."""
        width, height = self.get_screen_size()
        return 0 <= x < width and 0 <= y < height
    
    def alert(self, message: str, title: str = "Alert"):
        """Show alert dialog."""
        pyautogui.alert(text=message, title=title)
    
    def confirm(self, message: str, title: str = "Confirm") -> bool:
        """
        Show confirmation dialog.
        
        Returns:
            True if OK, False if Cancel
        """
        result = pyautogui.confirm(text=message, title=title)
        return result == 'OK'
    
    def prompt(self, message: str, title: str = "Input", default: str = "") -> Optional[str]:
        """
        Show input prompt.
        
        Returns:
            Input string or None if cancelled
        """
        return pyautogui.prompt(text=message, title=title, default=default)


class AutomationSequence:
    """Helper class for creating automation sequences."""
    
    def __init__(self):
        self.controller = AutomationController()
        self.actions = []
    
    def add_click(self, x: int, y: int, button: str = 'left', delay: float = 0.5):
        """Add click action to sequence."""
        self.actions.append(('click', {'x': x, 'y': y, 'button': button, 'delay': delay}))
        return self
    
    def add_type(self, text: str, delay: float = 0.5):
        """Add type action to sequence."""
        self.actions.append(('type', {'text': text, 'delay': delay}))
        return self
    
    def add_hotkey(self, *keys, delay: float = 0.5):
        """Add hotkey action to sequence."""
        self.actions.append(('hotkey', {'keys': keys, 'delay': delay}))
        return self
    
    def add_wait(self, seconds: float):
        """Add wait action to sequence."""
        self.actions.append(('wait', {'seconds': seconds}))
        return self
    
    def add_screenshot(self, save_path: str, region: Optional[Tuple[int, int, int, int]] = None):
        """Add screenshot action to sequence."""
        self.actions.append(('screenshot', {'path': save_path, 'region': region}))
        return self
    
    def execute(self):
        """Execute the sequence."""
        print(f"Executing {len(self.actions)} actions...")
        
        for i, (action_type, params) in enumerate(self.actions):
            print(f"Action {i+1}/{len(self.actions)}: {action_type}")
            
            if action_type == 'click':
                self.controller.click(params['x'], params['y'], button=params['button'])
                time.sleep(params['delay'])
            
            elif action_type == 'type':
                self.controller.type_text(params['text'])
                time.sleep(params['delay'])
            
            elif action_type == 'hotkey':
                self.controller.hotkey(*params['keys'])
                time.sleep(params['delay'])
            
            elif action_type == 'wait':
                time.sleep(params['seconds'])
            
            elif action_type == 'screenshot':
                img = self.controller.screenshot(region=params['region'])
                img.save(params['path'])
                print(f"  Screenshot saved to {params['path']}")
        
        print("Sequence complete!")
    
    def clear(self):
        """Clear all actions."""
        self.actions = []
