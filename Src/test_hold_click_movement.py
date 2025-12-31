"""Test script for hold-click and move mouse behavior."""
import pyautogui
import time
import argparse


def test_hold_and_move(positions, delay=0.5):
    """
    Test holding left click and moving through positions.
    
    Args:
        positions: List of (x, y) tuples
        delay: Delay between movements in seconds
    """
    print(f"\n{'='*60}")
    print("Testing Hold-Click and Move")
    print(f"{'='*60}\n")
    
    print(f"Will move through {len(positions)} positions:")
    for i, (x, y) in enumerate(positions):
        print(f"  {i+1}. ({x}, {y})")
    
    print(f"\nStarting in 3 seconds...")
    time.sleep(3)
    
    for i, (x, y) in enumerate(positions):
        if i == 0:
            # First position - start holding
            print(f"\n[Position {i+1}] Starting hold at ({x}, {y})")
            pyautogui.mouseDown(x, y)
        else:
            # Subsequent positions - move while holding
            print(f"[Position {i+1}] Moving to ({x}, {y}) while holding")
            pyautogui.moveTo(x, y, duration=0.3)
        
        # Wait at this position
        print(f"  -> Staying for {delay}s...")
        time.sleep(delay)
    
    # Release mouse
    print(f"\nReleasing mouse")
    pyautogui.mouseUp()
    
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test hold-click and move behavior")
    parser.add_argument("--positions", nargs='+', type=str,
                       help="Positions as 'x,y' pairs (e.g., '100,200 150,250 200,300')")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay at each position in seconds (default: 0.5)")
    parser.add_argument("--default", action="store_true",
                       help="Use default test positions (5 positions in a line)")
    args = parser.parse_args()
    
    if args.default or not args.positions:
        # Default: 5 positions for training stats (speed, stamina, power, guts, wit)
        # Based on actual game UI coordinates
        print("Using default test positions (training stat buttons)")
        positions = [
            (350, 870),  # speed
            (450, 870),  # stamina
            (550, 870),  # power
            (650, 870),  # guts
            (750, 870),  # wit
        ]
    else:
        # Parse positions from arguments
        positions = []
        for pos_str in args.positions:
            x, y = map(int, pos_str.split(','))
            positions.append((x, y))
    
    # Safety: Enable failsafe
    pyautogui.FAILSAFE = True
    print("\nFailsafe enabled: Move mouse to top-left corner to abort\n")
    
    # Run test
    test_hold_and_move(positions, args.delay)


if __name__ == "__main__":
    main()
