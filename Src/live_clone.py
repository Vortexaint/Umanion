"""Live preview controller (UI) for continuous OCR and overlay."""
import threading
import time
import numpy as np
from PIL import Image
try:
    import tkinter as tk
    from PIL import ImageTk
except Exception:
    tk = None
    ImageTk = None

from ocr import OCR
from capture import capture_primary_monitor


class LiveCloneController:
    """Live preview controller: capture, OCR and overlay matches."""

    def __init__(self, find_text=None, scale=0.5, interval=700, on_exit=None):
        self.find_text = find_text
        self.scale = float(scale) if scale else 1.0
        self.interval = int(interval)
        self._thread = None
        self._stop_event = threading.Event()
        self._running_event = threading.Event()
        self._root = None
        self._on_exit = on_exit
        self._last_time = None
        self._fps = 0.0
        self._fps_alpha = 0.22

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running_event.wait(timeout=3)

    def _run(self):
        if tk is None or ImageTk is None:
            print('Live mode requires tkinter and Pillow.ImageTk.\nInstall tkinter (bundle with Python) and Pillow.')
            return

        ocr = OCR()
        state = {'scanning': True}

        root = tk.Tk()
        self._root = root
        root.title('Uma - Live Clone')
        canvas = tk.Canvas(root, bg='black')
        canvas.pack(fill='both', expand=True)

        status_item = {'rect': None, 'text': None}

        def update_status_display():
            try:
                status = 'ON' if state['scanning'] else 'OFF'
                color = 'lime' if state['scanning'] else 'red'
                try:
                    w = max(160, canvas.winfo_width())
                except Exception:
                    w = 160
                try:
                    if status_item['rect'] is not None:
                        canvas.delete(status_item['rect'])
                    if status_item['text'] is not None:
                        canvas.delete(status_item['text'])
                except Exception:
                    pass
                status_item['rect'] = canvas.create_rectangle(w-140, 6, w-6, 36, fill='black', outline='')
                status_item['text'] = canvas.create_text(w-134, 10, anchor='nw', text=f'Scanning: {status}', fill=color)
            except Exception:
                pass

        def toggle_scanning(event=None):
            state['scanning'] = not state['scanning']
            status = 'ON' if state['scanning'] else 'OFF'
            print(f'Live scanning toggled: {status}')
            try:
                update_status_display()
            except Exception:
                pass

        root.bind('<s>', toggle_scanning)
        root.bind('<S>', toggle_scanning)

        photo_ref = {'img': None}

        def update_frame():
            if self._stop_event.is_set():
                try:
                    root.destroy()
                except Exception:
                    pass
                return

            try:
                img = capture_primary_monitor()
            except Exception:
                root.after(self.interval, update_frame)
                return

            if self.scale != 1.0:
                w, h = img.size
                disp_size = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
                disp_img = img.resize(disp_size, Image.BILINEAR)
            else:
                disp_img = img

            tkimg = ImageTk.PhotoImage(disp_img)
            photo_ref['img'] = tkimg
            canvas.config(width=disp_img.size[0], height=disp_img.size[1])
            canvas.delete('all')
            canvas.create_image(0, 0, anchor='nw', image=tkimg)

            now = time.time()
            if self._last_time is None:
                self._last_time = now
            else:
                dt = now - self._last_time
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self._fps = (self._fps_alpha * inst_fps) + ((1.0 - self._fps_alpha) * self._fps)
                self._last_time = now

            try:
                fps_text = f"{self._fps:0.1f} FPS"
                canvas.create_rectangle(6, 6, 120, 28, fill='black', outline='')
                canvas.create_text(10, 10, anchor='nw', text=fps_text, fill='lime')
            except Exception:
                pass

            if state['scanning'] and self.find_text:
                try:
                    np_img = np.array(disp_img)
                    words = ocr.extract_words_with_boxes(np_img)
                    q_lower = self.find_text.lower()
                    matches = []
                    for wbox in words:
                        if q_lower in wbox['text'].lower():
                            left = wbox['left']
                            top = wbox['top']
                            right = left + wbox.get('width', 0)
                            bottom = top + wbox.get('height', 0)
                            matches.append((left, top, right, bottom))

                    for (l, t, r, b) in matches:
                        canvas.create_rectangle(l, t, r, b, outline='lime', width=3)
                except Exception:
                    try:
                        text = ocr.extract_text(np.array(disp_img))
                        if self.find_text.lower() in text.lower():
                            canvas.create_rectangle(10, 10, 200, 60, outline='lime', width=3)
                            canvas.create_text(15, 15, anchor='nw', text=f'Found: {self.find_text}', fill='lime')
                    except Exception:
                        pass

            try:
                update_status_display()
            except Exception:
                pass

            root.after(self.interval, update_frame)

        root.after(10, update_frame)
        self._running_event.set()
        try:
            root.mainloop()
        finally:
            self._running_event.clear()
            try:
                if self._on_exit:
                    try:
                        self._on_exit(self)
                    except Exception:
                        pass
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()
        try:
            if self._root is not None:
                try:
                    self._root.after(0, self._root.destroy)
                except Exception:
                    try:
                        self._root.destroy()
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if self._thread is not None and threading.current_thread() is not self._thread:
                self._thread.join(timeout=2)
        except Exception:
            pass
        self._thread = None

    def run_foreground(self):
        """Run the UI loop in the current thread (blocking)."""
        self._stop_event.clear()
        try:
            self._run()
        finally:
            try:
                self._running_event.clear()
            except Exception:
                pass

    def request_close(self):
        """Request the UI to close without joining threads."""
        self._stop_event.set()
        try:
            if self._root is not None:
                try:
                    if threading.current_thread() is self._thread:
                        self._root.destroy()
                    else:
                        self._root.after(0, self._root.destroy)
                except Exception:
                    try:
                        self._root.after(0, self._root.destroy)
                    except Exception:
                        pass
        except Exception:
            pass


def start_live_clone(find_text=None, scale=0.5, interval=700):
    ctrl = LiveCloneController(find_text=find_text, scale=scale, interval=interval)
    ctrl.start()
    return ctrl
