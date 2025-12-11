"""Main launcher for Uma.

Controller mode; does not start the live clone automatically.
Press F1 to toggle the live clone (requires the `keyboard` package).
"""

import argparse
import time
import threading


def main():
	parser = argparse.ArgumentParser(description='Uma main controller (hotkey + live toggle)')
	parser.add_argument('--find-text', dest='find_text', help='Text to search for in live overlay', default=None)
	parser.add_argument('--scale', type=float, default=0.5, help='Preview scale for clone')
	parser.add_argument('--interval', type=int, default=700, help='Update interval in ms for live preview')
	parser.add_argument('--start-live', action='store_true', help='Start live clone immediately')
	args = parser.parse_args()

	try:
		from live_clone import LiveCloneController
	except Exception as e:
		print('Failed to import LiveCloneController from live_clone:', e)
		return

	try:
		import keyboard
	except Exception:
		keyboard = None

	controller = {'inst': None}

	def start_clone():
		if controller['inst'] is None:
			def _on_exit_callback(c):
				try:
					if controller['inst'] is c:
						controller['inst'] = None
						print('Live clone stopped (window closed)')
				except Exception:
					pass

			ctrl = LiveCloneController(find_text=args.find_text, scale=args.scale, interval=args.interval, on_exit=_on_exit_callback)
			ctrl.start()
			controller['inst'] = ctrl
			print('Live clone started')

	def stop_clone():
		if controller['inst'] is not None:
			try:
				controller['inst'].stop()
			except Exception:
				pass
			controller['inst'] = None
			print('Live clone stopped')

	def toggle_clone():
		if controller['inst'] is None:
			start_clone()
		else:
			stop_clone()

	if keyboard is not None:
		try:
			keyboard.add_hotkey('F1', toggle_clone)
			print('Global hotkey F1 registered (press F1 to toggle live clone)')
		except Exception:
			keyboard = None

	if keyboard is None:
		print('Module `keyboard` not available. Install with: pip install keyboard')
		print('Console controls: type "t" + Enter to toggle, "q" + Enter to quit')

		def console_loop():
			while True:
				try:
					s = input().strip().lower()
				except EOFError:
					break
				if s == 't':
					toggle_clone()
				elif s == 'q':
					break

		console_thread = threading.Thread(target=console_loop, daemon=True)
		console_thread.start()

	if args.start_live:
		start_clone()

	try:
		while True:
			time.sleep(0.5)
	except KeyboardInterrupt:
		print('\nShutting down...')
	finally:
		stop_clone()


if __name__ == '__main__':
	main()
