"""Screen capture system for monitoring and capturing live screen data."""
import mss
import mss.tools
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Optional
import pygetwindow as gw


class ScreenCapture:
    """Handles screen capture from monitors."""
    
    def __init__(self):
        self.sct = mss.mss()
        self.monitors = self.sct.monitors
        
    def get_monitor_count(self) -> int:
        """Get the number of available monitors."""
        return len(self.monitors) - 1  # Exclude the first 'all monitors' entry
    
    def get_monitor_info(self, monitor_num: int = 1) -> dict:
        """Get information about a specific monitor."""
        if monitor_num < 1 or monitor_num >= len(self.monitors):
            raise ValueError(f"Monitor {monitor_num} not found. Available: 1-{self.get_monitor_count()}")
        return self.monitors[monitor_num]
    
    def get_primary_monitor(self) -> int:
        """Get the primary monitor number (usually 1, but can vary)."""
        # Monitor 1 is typically primary, but let's verify
        # The primary monitor usually has top-left at (0,0) or close to it
        for i in range(1, len(self.monitors)):
            monitor = self.monitors[i]
            if monitor['left'] == 0 and monitor['top'] == 0:
                return i
        return 1  # Default to monitor 1
    
    def list_all_monitors(self):
        """Print information about all available monitors."""
        print(f"\nAvailable Monitors: {self.get_monitor_count()}")
        print("-" * 60)
        for i in range(1, len(self.monitors)):
            mon = self.monitors[i]
            is_primary = "(PRIMARY)" if (mon['left'] == 0 and mon['top'] == 0) else ""
            print(f"Monitor {i} {is_primary}:")
            print(f"  Position: ({mon['left']}, {mon['top']})")
            print(f"  Size: {mon['width']}x{mon['height']}")
            print()
    
    def capture_monitor(self, monitor_num: int = 1) -> Image.Image:
        """
        Capture screenshot from specified monitor.
        
        Args:
            monitor_num: Monitor number (1 for primary, 2+ for additional)
            
        Returns:
            PIL Image of the captured screen
        """
        monitor = self.monitors[monitor_num]
        screenshot = self.sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img
    
    def capture_region(self, x: int, y: int, width: int, height: int, monitor_num: int = 1) -> Image.Image:
        """
        Capture a specific region of the screen.
        
        Args:
            x, y: Top-left coordinates
            width, height: Region dimensions
            monitor_num: Monitor number
            
        Returns:
            PIL Image of the captured region
        """
        monitor = {
            "left": x,
            "top": y,
            "width": width,
            "height": height,
            "mon": monitor_num
        }
        screenshot = self.sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img
    
    def capture_window(self, window_title: str) -> Optional[Image.Image]:
        """
        Capture a specific window by title.
        
        Args:
            window_title: Title of the window to capture
            
        Returns:
            PIL Image or None if window not found
        """
        try:
            windows = gw.getWindowsWithTitle(window_title)
            if not windows:
                return None
            
            window = windows[0]
            if window.isMinimized:
                window.restore()
            
            # Capture the window region
            return self.capture_region(
                window.left,
                window.top,
                window.width,
                window.height
            )
        except Exception as e:
            print(f"Error capturing window: {e}")
            return None
    
    def save_screenshot(self, filename: str, monitor_num: int = 1):
        """Save screenshot to file."""
        img = self.capture_monitor(monitor_num)
        img.save(filename)
        print(f"Screenshot saved to {filename}")
    
    def get_numpy_array(self, monitor_num: int = 1) -> np.ndarray:
        """Get monitor capture as numpy array."""
        img = self.capture_monitor(monitor_num)
        return np.array(img)


class LiveMonitor:
    """Live monitoring system for continuous screen scanning."""
    
    def __init__(self, monitor_num: int = 1, fps: int = 10):
        """
        Initialize live monitor.
        
        Args:
            monitor_num: Monitor to scan
            fps: Frames per second for scanning
        """
        self.capture = ScreenCapture()
        self.monitor_num = monitor_num
        self.fps = fps
        self.is_running = False
        
    def start_monitoring(self, callback):
        """
        Start live monitoring with callback function.
        
        Args:
            callback: Function to call with each frame (PIL Image)
        """
        import time
        
        self.is_running = True
        frame_time = 1.0 / self.fps
        
        print(f"Starting live monitor on Monitor {self.monitor_num} at {self.fps} FPS")
        print("Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                start = time.time()
                
                # Capture frame
                frame = self.capture.capture_monitor(self.monitor_num)
                
                # Process frame with callback
                callback(frame)
                
                # Maintain FPS
                elapsed = time.time() - start
                sleep_time = max(0, frame_time - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            self.is_running = False
    
    def stop_monitoring(self):
        """Stop live monitoring."""
        self.is_running = False


def count_pixels_of_color(color_rgb=(117, 117, 117), region: Optional[Tuple[int, int, int, int]] = None, tolerance: int = 2, monitor_num: int = 1) -> int:
    """
    Count pixels in a screen region that match a target RGB color within a tolerance.

    Args:
        color_rgb: Target color as (R, G, B).
        region: Tuple (left, top, right, bottom) in screen coordinates.
        tolerance: +/- tolerance per channel.
        monitor_num: Monitor number to capture from.

    Returns:
        Number of pixels matching the color in the region, or -1 on error.
    """
    if region is None:
        return -1

    left, top, right, bottom = region
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        return -1

    sc = ScreenCapture()
    try:
        img = sc.capture_region(left, top, width, height, monitor_num=monitor_num)
    except Exception:
        return -1

    screen = np.array(img)

    color = np.array(color_rgb, dtype=np.uint8)
    color_min = np.clip(color - tolerance, 0, 255)
    color_max = np.clip(color + tolerance, 0, 255)

    # cv2 expects BGR order; our array from PIL is RGB, so convert
    screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

    dst = cv2.inRange(screen_bgr, color_min[::-1], color_max[::-1])
    pixel_count = int(cv2.countNonZero(dst))
    return pixel_count


class EnergyReader:
    """Utility to read an energy bar on screen by counting missing-energy pixels.

    The bar region defaults to x=442..676 at y=136 with a +/-2 px vertical tolerance.
    Missing energy is represented by gray (117,117,117).
    """

    # default region (left, top, right, bottom)
    DEFAULT_REGION = (442 - 2, 136 - 2, 676 + 2, 136 + 2)
    MISSING_COLOR = (117, 117, 117)

    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None, tolerance: int = 2, monitor_num: int = 1):
        self.region = region or self.DEFAULT_REGION
        self.tolerance = tolerance
        self.monitor_num = monitor_num

    def read(self) -> dict:
        """Return pixel counts and percentage filled for the energy bar.

        Returns:
            dict with keys: `missing_pixels`, `total_pixels`, `filled_pixels`, `percent_filled`.
        """
        left, top, right, bottom = self.region
        width = max(0, right - left)
        height = max(0, bottom - top)
        total = width * height

        missing = count_pixels_of_color(self.MISSING_COLOR, region=self.region, tolerance=self.tolerance, monitor_num=self.monitor_num)
        if missing < 0 or total == 0:
            return {"missing_pixels": -1, "total_pixels": total, "filled_pixels": -1, "percent_filled": -1.0}

        filled = total - missing
        percent = (filled / total) * 100.0
        # energy as integer 0-100
        energy = int(round(percent))
        energy = max(0, min(100, energy))
        return {"missing_pixels": missing, "total_pixels": total, "filled_pixels": filled, "percent_filled": percent, "energy": energy}


def print_current_energy(region: Optional[Tuple[int, int, int, int]] = None, tolerance: int = 2, monitor_num: int = 1) -> int:
    """Capture the energy bar and print a concise debug line.

    Returns the integer energy (0-100) or -1 on failure.
    """
    er = EnergyReader(region=region, tolerance=tolerance, monitor_num=monitor_num)
    info = er.read()
    energy = info.get("energy", -1)
    print(f"Energy: {energy}% — details: {info}")
    return energy


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Debug energy reader")
    parser.add_argument("--once", action="store_true", help="Print energy once")
    parser.add_argument("--loop", type=int, default=0, help="Print energy N times with 1s interval")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor number to capture from")
    args = parser.parse_args()

    if args.once or args.loop == 0:
        print_current_energy(region=EnergyReader.DEFAULT_REGION, tolerance=2, monitor_num=args.monitor)
    else:
        for i in range(args.loop):
            print_current_energy(region=EnergyReader.DEFAULT_REGION, tolerance=2, monitor_num=args.monitor)
            if i + 1 < args.loop:
                time.sleep(1.0)
