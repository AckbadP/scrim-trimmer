"""
gui.py - Tkinter GUI for the EVE AT Practice Trimmer.

Prerequisites:
  sudo apt-get install python3-tk
  pip install tkinterdnd2

Launch:
  python src/gui.py
"""

import argparse
import os
import sys
import threading

import psutil


def _resource_path(relative_to_src: str) -> str:
    """Return absolute path to a bundled resource, works for dev and PyInstaller."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_to_src)

import config as cfg

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, font as tkfont
except ImportError as e:
    print(f"Error: {e}")
    print("Install prerequisites: sudo apt-get install python3-tk && pip install tkinterdnd2")
    sys.exit(1)

try:
    import cv2
    from PIL import Image, ImageTk
except ImportError as e:
    print(f"Error: {e}")
    print("Install prerequisites: pip install opencv-python pillow")
    sys.exit(1)

from frame_extractor import get_video_duration
import main as pipeline
import native_dialogs


VALID_EXTENSIONS = {".mp4", ".mkv"}
THUMB_W, THUMB_H = 320, 180

_DEFAULT_REGION = [0.0, 0.35, 0.15, 1.0]  # [x1, y1, x2, y2] as fractions


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("EVE AT Practice Trimmer")
        self.resizable(True, True)

        self.video_path: str | None = None
        self._thumb_ref = None  # prevent GC of PhotoImage
        self._thumb_pil: Image.Image | None = None  # source image for re-rendering on resize

        # Canvas state
        self._canvas_img_id = None
        self._canvas_rect_id = None
        self._drag_start = None
        # Letterboxed image rect within the canvas: (x_offset, y_offset, width, height)
        self._img_rect: tuple[int, int, int, int] | None = None

        self._conf = cfg.load_config()
        _conf = self._conf
        self.default_video_dir_var = tk.StringVar(value=_conf.get("video_dir", ""))
        self.default_log_dir_var = tk.StringVar(value=_conf.get("log_dir", ""))
        self.default_output_dir_var = tk.StringVar(value=_conf.get("output_dir", ""))
        self.default_chapters_dir_var = tk.StringVar(value=_conf.get("chapters_dir", ""))

        raw = _conf.get("chat_region", _DEFAULT_REGION)
        self._chat_region: list[float] = list(raw) if len(raw) == 4 else list(_DEFAULT_REGION)

        # Auto-save config whenever any default directory changes
        self.default_video_dir_var.trace_add("write", self._save_config)
        self.default_log_dir_var.trace_add("write", self._save_config)
        self.default_output_dir_var.trace_add("write", self._save_config)
        self.default_chapters_dir_var.trace_add("write", self._save_config)
        self.show_chapters_popup_var = tk.BooleanVar(value=bool(_conf.get("show_chapters_popup", True)))
        self.show_chapters_popup_var.trace_add("write", self._save_config)
        self.close_on_complete_var = tk.BooleanVar(value=bool(_conf.get("close_on_complete", False)))
        self.close_on_complete_var.trace_add("write", self._save_config)

        self._build_ui()

        # Auto-load most recent video if a default video directory is configured
        video_dir = os.path.expanduser(self.default_video_dir_var.get().strip())
        if video_dir and os.path.isdir(video_dir):
            latest = self._find_latest_video(video_dir)
            if latest:
                self._load_video(latest)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_frame = ttk.Frame(self, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Drop zone ---
        self.drop_frame = ttk.LabelFrame(main_frame, text="Video File", padding=12)
        self.drop_frame.pack(fill=tk.X)

        self.drop_hint = ttk.Label(
            self.drop_frame,
            text="Drag a video file here",
            font=("TkDefaultFont", 13),
            anchor="center",
        )
        self.drop_hint.pack(fill=tk.X, pady=(8, 6))

        ttk.Button(self.drop_frame, text="Browse...", command=self._browse).pack()

        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind("<<Drop>>", self._on_drop)

        # --- Preview (hidden until file loaded) ---
        self.preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding=8)

        # Left column: canvas + hint
        canvas_col = ttk.Frame(self.preview_frame)
        canvas_col.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=True)

        _bg = ttk.Style().lookup("TFrame", "background") or self.cget("bg")
        self.thumb_canvas = tk.Canvas(
            canvas_col,
            width=THUMB_W, height=THUMB_H,
            bg=_bg, bd=0,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.thumb_canvas.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            canvas_col,
            text="Drag to select chat region",
            font=("TkDefaultFont", 8),
            foreground="#888",
        ).pack(pady=(2, 0))

        self.thumb_canvas.bind("<ButtonPress-1>", self._on_region_press)
        self.thumb_canvas.bind("<B1-Motion>", self._on_region_drag)
        self.thumb_canvas.bind("<ButtonRelease-1>", self._on_region_release)
        self.thumb_canvas.bind("<Configure>", self._on_canvas_resize)

        self.info_label = ttk.Label(canvas_col, text="", justify=tk.CENTER)
        self.info_label.pack(pady=(2, 0))

        # Pre-pack then hide so later .pack(after=drop_frame) places it correctly
        self.preview_frame.pack(after=self.drop_frame, fill=tk.BOTH, expand=True, pady=(10, 0))
        self.preview_frame.pack_forget()

        # --- Options ---
        self.opts_frame = ttk.LabelFrame(main_frame, text="Options", padding=8)
        self.opts_frame.pack(fill=tk.X, pady=(10, 0))

        saved_out = self.default_output_dir_var.get()
        self.output_var = tk.StringVar(value=saved_out if saved_out else "out")
        self.t0_var = tk.StringVar()
        self.chat_log_var = tk.StringVar()
        self.verbose_var = tk.BooleanVar(value=False)
        self.force_python_var = tk.BooleanVar(value=False)
        self.force_ocr_var = tk.BooleanVar(value=False)
        _cpu = os.cpu_count() or 4
        self.threads_var = tk.IntVar(value=self._conf.get("threads") or _cpu)
        self.ram_cap_var = tk.IntVar(value=self._conf.get("ram_cap_gb") or 10)

        notebook = ttk.Notebook(self.opts_frame)
        notebook.pack(fill=tk.X)

        # --- Main tab ---
        main_tab = ttk.Frame(notebook, padding=(4, 6))
        main_tab.columnconfigure(1, weight=1)
        notebook.add(main_tab, text="Main")

        # output dir
        out_frame = ttk.Frame(main_tab)
        out_frame.grid(row=0, column=1, sticky=tk.EW, pady=2)
        out_frame.columnconfigure(0, weight=1)
        ttk.Label(main_tab, text="Output dir").grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 8))
        ttk.Entry(out_frame, textvariable=self.output_var).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(out_frame, text="Browse", width=7,
                   command=self._browse_output).grid(row=0, column=1, padx=(4, 0))

        # chat-log
        ttk.Label(main_tab, text="Chat log").grid(row=1, column=0, sticky=tk.W, pady=2, padx=(0, 8))
        cl_frame = ttk.Frame(main_tab)
        cl_frame.grid(row=1, column=1, sticky=tk.EW, pady=2)
        cl_frame.columnconfigure(0, weight=1)
        ttk.Entry(cl_frame, textvariable=self.chat_log_var).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(cl_frame, text="Browse", width=7,
                   command=self._browse_chatlog).grid(row=0, column=1, padx=(4, 0))

        # --- Advanced tab ---
        adv_tab = ttk.Frame(notebook, padding=(4, 6))
        adv_tab.columnconfigure(1, weight=1)
        notebook.add(adv_tab, text="Advanced")

        # defaults sub-section
        defaults_lf = ttk.LabelFrame(adv_tab, text="Default Directories", padding=6)
        defaults_lf.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        defaults_lf.columnconfigure(1, weight=1)

        def _defaults_row(row, label, var, browse_cmd):
            ttk.Label(defaults_lf, text=label).grid(
                row=row, column=0, sticky=tk.W, pady=2, padx=(0, 8))
            f = ttk.Frame(defaults_lf)
            f.grid(row=row, column=1, sticky=tk.EW, pady=2)
            f.columnconfigure(0, weight=1)
            e = ttk.Entry(f, textvariable=var)
            e.grid(row=0, column=0, sticky=tk.EW)
            e.bind("<FocusOut>", lambda _: self._save_config())
            ttk.Button(f, text="Browse", width=7, command=browse_cmd).grid(
                row=0, column=1, padx=(4, 0))

        _defaults_row(0, "Video directory", self.default_video_dir_var, self._browse_default_video_dir)
        _defaults_row(1, "Log directory", self.default_log_dir_var, self._browse_default_log_dir)
        _defaults_row(2, "Output directory", self.default_output_dir_var, self._browse_default_output_dir)
        _defaults_row(3, "Chapters file dir", self.default_chapters_dir_var, self._browse_default_chapters_dir)

        # t0
        ttk.Label(adv_tab, text="Recording start time (UTC)").grid(
            row=1, column=0, sticky=tk.W, pady=(4, 0), padx=(0, 8))
        t0_entry = ttk.Entry(adv_tab, textvariable=self.t0_var, width=14)
        t0_entry.grid(row=1, column=1, sticky=tk.W, pady=(4, 0))
        t0_entry.insert(0, "HH:MM:SS (optional)")
        t0_entry.bind("<FocusIn>", lambda e: t0_entry.delete(0, tk.END)
                      if self.t0_var.get() == "HH:MM:SS (optional)" else None)
        t0_entry.bind("<FocusOut>", lambda e: t0_entry.insert(0, "HH:MM:SS (optional)")
                      if not self.t0_var.get() else None)
        ttk.Label(adv_tab, text="EVE game clock (UTC) when the video started — used to sync\n"
                                "chat log timestamps. Auto-detected if left blank.",
                  foreground="#888", font=("TkDefaultFont", 8)).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 4), padx=(0, 8))

        # threads
        ttk.Label(adv_tab, text="Threads").grid(row=3, column=0, sticky=tk.W, pady=2, padx=(0, 8))
        threads_frame = ttk.Frame(adv_tab)
        threads_frame.grid(row=3, column=1, sticky=tk.W, pady=2)
        ttk.Spinbox(threads_frame, from_=1, to=64, textvariable=self.threads_var,
                    width=5, command=self._save_config).grid(row=0, column=0)
        ttk.Label(threads_frame, text=f"(CPU cores: {os.cpu_count() or '?'})",
                  foreground="#888").grid(row=0, column=1, padx=(6, 0))

        # ram cap
        ttk.Label(adv_tab, text="RAM cap (GB)").grid(row=4, column=0, sticky=tk.W, pady=2, padx=(0, 8))
        ram_frame = ttk.Frame(adv_tab)
        ram_frame.grid(row=4, column=1, sticky=tk.W, pady=2)
        ttk.Spinbox(ram_frame, from_=1, to=256, textvariable=self.ram_cap_var,
                    width=5, command=self._save_config).grid(row=0, column=0)
        _total_ram = round(psutil.virtual_memory().total / 1024 ** 3)
        ttk.Label(ram_frame, text=f"GB max frames in memory (total: {_total_ram} GB)",
                  foreground="#888").grid(row=0, column=1, padx=(6, 0))

        # verbose
        ttk.Label(adv_tab, text="verbose").grid(row=5, column=0, sticky=tk.W, pady=2, padx=(0, 8))
        ttk.Checkbutton(adv_tab, variable=self.verbose_var).grid(row=5, column=1, sticky=tk.W, pady=2)

        # force Python clipper
        ttk.Label(adv_tab, text="Force Python clipper").grid(
            row=6, column=0, sticky=tk.W, pady=2, padx=(0, 8))
        ttk.Checkbutton(
            adv_tab, variable=self.force_python_var,
            command=self._on_force_python_toggle,
        ).grid(row=6, column=1, sticky=tk.W, pady=2)

        # force OCR pipeline
        ttk.Label(adv_tab, text="Force OCR pipeline").grid(
            row=7, column=0, sticky=tk.W, pady=2, padx=(0, 8))
        ttk.Checkbutton(adv_tab, variable=self.force_ocr_var).grid(
            row=7, column=1, sticky=tk.W, pady=2)

        # chapter timestamps popup
        ttk.Label(adv_tab, text="Show timestamps popup").grid(
            row=8, column=0, sticky=tk.W, pady=2, padx=(0, 8))
        ttk.Checkbutton(adv_tab, variable=self.show_chapters_popup_var).grid(
            row=8, column=1, sticky=tk.W, pady=2)

        # close on complete
        ttk.Label(adv_tab, text="Close when done").grid(
            row=9, column=0, sticky=tk.W, pady=2, padx=(0, 8))
        ttk.Checkbutton(adv_tab, variable=self.close_on_complete_var).grid(
            row=9, column=1, sticky=tk.W, pady=2)

        def _on_tab_changed(event=None):
            tab = notebook.nametowidget(notebook.select())
            new_nb_height = tab.winfo_reqheight()
            delta = new_nb_height - notebook.winfo_height()
            if delta != 0:
                self.geometry(f"{self.winfo_width()}x{self.winfo_height() + delta}")
            notebook.configure(height=new_nb_height)

        notebook.bind("<<NotebookTabChanged>>", _on_tab_changed)

        def _apply_initial_tab_size():
            self.update_idletasks()
            tab = notebook.nametowidget(notebook.select())
            notebook.configure(height=tab.winfo_reqheight())

        self.after_idle(_apply_initial_tab_size)

        # --- Reset / Help buttons ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=(12, 4))

        ttk.Button(btn_frame, text="Reset", command=self._reset, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Help", command=self._show_help, width=10).pack(side=tk.LEFT, padx=6)

        # --- Status bar ---
        self.status_label = ttk.Label(main_frame, text="Ready", anchor=tk.W,
                                      relief=tk.SUNKEN, padding=(4, 2))
        self.status_label.pack(fill=tk.X, pady=(4, 0))

        # --- Run button (bottom) ---
        run_frame = ttk.Frame(main_frame)
        run_frame.pack(fill=tk.X, pady=(8, 4))

        self.run_btn = ttk.Button(run_frame, text="Run", command=self._run, width=20,
                                  padding=(0, 8))
        self.run_btn.pack(fill=tk.X, padx=6)

    # ------------------------------------------------------------------
    # File acquisition
    # ------------------------------------------------------------------

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")
        self._load_video(path)

    def _browse(self):
        path = native_dialogs.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.mkv"), ("All files", "*.*")],
        )
        if path:
            self._load_video(path)

    def _browse_output(self):
        d = native_dialogs.askdirectory(title="Select output directory")
        if d:
            self.output_var.set(d)

    def _browse_chatlog(self):
        paths = native_dialogs.askopenfilenames(
            title="Select EVE chat log(s)",
            filetypes=[("Log files", "*.txt *.log"), ("All files", "*.*")],
        )
        if paths:
            self.chat_log_var.set(";".join(paths))

    def _browse_default_video_dir(self):
        d = native_dialogs.askdirectory(title="Select default video directory")
        if d:
            self.default_video_dir_var.set(d)
            latest = self._find_latest_video(d)
            if latest:
                self._load_video(latest)

    def _browse_default_log_dir(self):
        d = native_dialogs.askdirectory(title="Select default log directory")
        if d:
            self.default_log_dir_var.set(d)

    def _browse_default_output_dir(self):
        d = native_dialogs.askdirectory(title="Select default output directory")
        if d:
            self.default_output_dir_var.set(d)

    def _browse_default_chapters_dir(self):
        d = native_dialogs.askdirectory(title="Select default chapters file directory")
        if d:
            self.default_chapters_dir_var.set(d)

    # ------------------------------------------------------------------
    # Video loading + thumbnail
    # ------------------------------------------------------------------

    def _load_video(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in VALID_EXTENSIONS:
            messagebox.showerror("Invalid file",
                                 f"Unsupported file type '{ext}'.\nExpected .mp4 or .mkv.")
            return

        if not os.path.isfile(path):
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return

        self.video_path = path

        # Thumbnail
        pil = self._extract_thumbnail(path)
        if pil:
            self._update_canvas_image(pil)
        else:
            self._thumb_pil = None
            self.thumb_canvas.delete("all")
            self._canvas_img_id = None
            self._canvas_rect_id = None
            self.thumb_canvas.create_text(
                THUMB_W // 2, THUMB_H // 2, text="(no preview)", fill="white"
            )

        # Info label
        try:
            duration = get_video_duration(path)
            dur_str = self._fmt_duration(duration)
        except Exception:
            dur_str = "?"
        self.info_label.configure(
            text=f"{os.path.basename(path)}\n{dur_str}"
        )

        # Auto-fill output dir from default
        default_out = os.path.expanduser(self.default_output_dir_var.get().strip())
        if default_out:
            self.output_var.set(default_out)

        # Auto-select closest "Local" log — reveal first so errors are always visible
        self.preview_frame.pack(after=self.drop_frame, fill=tk.BOTH, expand=True, pady=(10, 0))
        self.run_btn.configure(state="normal")

        status_log = ""
        _log_dir_setting = self.default_log_dir_var.get().strip()
        default_log = os.path.expanduser(_log_dir_setting) if _log_dir_setting else cfg.default_log_dir()
        if default_log:
            if not os.path.isdir(default_log):
                status_log = f"  |  Log dir not found: {default_log}"
            else:
                try:
                    best = self._find_closest_log(path, default_log)
                    if best:
                        self.chat_log_var.set(best)
                        status_log = f"  |  Log: {os.path.basename(best)}"
                    else:
                        all_files = [
                            f for f in os.listdir(default_log)
                            if os.path.isfile(os.path.join(default_log, f))
                        ]
                        status_log = (
                            f"  |  No 'Local' file in {os.path.basename(default_log)}"
                            f" ({len(all_files)} files total)"
                        )
                except Exception as exc:
                    status_log = f"  |  Log error: {exc}"

        self._set_status(f"Loaded: {os.path.basename(path)}{status_log}")

    def _extract_thumbnail(self, path: str) -> Image.Image | None:
        try:
            cap = cv2.VideoCapture(path)
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.10))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Canvas region drawing & interaction
    # ------------------------------------------------------------------

    def _update_canvas_image(self, pil: Image.Image):
        """Store the source PIL image and render it at the current canvas size."""
        self._thumb_pil = pil
        self._redraw_canvas()

    def _redraw_canvas(self):
        """Re-render the stored PIL image letterboxed into the canvas, then redraw rect."""
        cw = self.thumb_canvas.winfo_width()
        ch = self.thumb_canvas.winfo_height()
        if self._thumb_pil is None or cw <= 1 or ch <= 1:
            return
        iw, ih = self._thumb_pil.size
        scale = min(cw / iw, ch / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self._img_rect = (ox, oy, dw, dh)

        resized = self._thumb_pil.resize((dw, dh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        self._thumb_ref = photo
        if self._canvas_img_id is None:
            self._canvas_img_id = self.thumb_canvas.create_image(
                ox, oy, anchor=tk.NW, image=photo
            )
        else:
            self.thumb_canvas.itemconfig(self._canvas_img_id, image=photo)
            self.thumb_canvas.coords(self._canvas_img_id, ox, oy)
        self._draw_region_rect()

    def _on_canvas_resize(self, event):
        self._redraw_canvas()

    def _draw_region_rect(self):
        """Draw/update the chat region rectangle mapped through the letterboxed image rect."""
        if self._img_rect is None:
            return
        ox, oy, dw, dh = self._img_rect
        x1, y1, x2, y2 = self._chat_region
        cx1, cy1 = ox + int(x1 * dw), oy + int(y1 * dh)
        cx2, cy2 = ox + int(x2 * dw), oy + int(y2 * dh)
        if self._canvas_rect_id is None:
            self._canvas_rect_id = self.thumb_canvas.create_rectangle(
                cx1, cy1, cx2, cy2, outline="#00ff00", width=2, dash=(4, 2)
            )
        else:
            self.thumb_canvas.coords(self._canvas_rect_id, cx1, cy1, cx2, cy2)
        self.thumb_canvas.tag_raise(self._canvas_rect_id)

    def _on_region_press(self, event):
        self._drag_start = (event.x, event.y)
        if self._canvas_rect_id is None:
            self._canvas_rect_id = self.thumb_canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="#00ff00", width=2, dash=(4, 2)
            )
        else:
            self.thumb_canvas.coords(
                self._canvas_rect_id, event.x, event.y, event.x, event.y
            )
        self.thumb_canvas.tag_raise(self._canvas_rect_id)

    def _on_region_drag(self, event):
        if self._drag_start is None or self._canvas_rect_id is None:
            return
        x0, y0 = self._drag_start
        self.thumb_canvas.coords(
            self._canvas_rect_id,
            min(x0, event.x), min(y0, event.y),
            max(x0, event.x), max(y0, event.y),
        )

    def _on_region_release(self, event):
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None

        if self._img_rect is None:
            return
        ox, oy, dw, dh = self._img_rect
        x1 = max(0.0, min(1.0, (min(x0, event.x) - ox) / dw))
        y1 = max(0.0, min(1.0, (min(y0, event.y) - oy) / dh))
        x2 = max(0.0, min(1.0, (max(x0, event.x) - ox) / dw))
        y2 = max(0.0, min(1.0, (max(y0, event.y) - oy) / dh))

        # Ignore tiny drags (accidental clicks)
        if x2 - x1 < 0.02 or y2 - y1 < 0.02:
            self._draw_region_rect()  # restore previous rect
            return

        self._chat_region = [x1, y1, x2, y2]
        self._draw_region_rect()
        self._save_config()

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    _FFMPEG_WARNING = (
        "You are enabling the Python/OpenCV clipper fallback.\n\n"
        "Compared to ffmpeg, this mode has significant drawbacks:\n"
        "  \u2022  Audio is stripped — the output video will be silent.\n"
        "  \u2022  Video is re-encoded — slower and may lose quality.\n\n"
        "It is strongly recommended to install ffmpeg instead:\n\n"
        "  Ubuntu / Debian:\n"
        "    sudo apt install ffmpeg\n\n"
        "  Windows:\n"
        "    1. Download from https://ffmpeg.org/download.html\n"
        "    2. Extract the archive and add the bin\\ folder to your PATH.\n\n"
        "Continue with the Python clipper anyway?"
    )

    def _on_force_python_toggle(self):
        if self.force_python_var.get():
            proceed = messagebox.askokcancel(
                "Warning: Python clipper has limitations",
                self._FFMPEG_WARNING,
                icon="warning",
            )
            if not proceed:
                self.force_python_var.set(False)

    def _run(self):
        if not self.video_path:
            return

        t0_raw = self.t0_var.get().strip()
        if t0_raw in ("", "HH:MM:SS (optional)"):
            t0_raw = None

        chat_log_raw = self.chat_log_var.get().strip()
        chat_logs = [p for p in chat_log_raw.split(";") if p] if chat_log_raw else None

        if not chat_logs:
            _warning_file = _resource_path("no_log_warning.txt")
            with open(_warning_file) as _f:
                _warning_text = _f.read()
            proceed = messagebox.askokcancel("No logs provided", _warning_text)
            if not proceed:
                return

        chapters_dir = os.path.expanduser(self.default_chapters_dir_var.get().strip()) or None
        args = argparse.Namespace(
            video=self.video_path,
            output=os.path.expanduser(self.output_var.get()) or "out",
            chat_region=list(self._chat_region),
            verbose=bool(self.verbose_var.get()),
            chat_logs=chat_logs,
            t0=t0_raw,
            threads=self.threads_var.get(),
            ram_cap_gb=self.ram_cap_var.get(),
            force_python_clipper=bool(self.force_python_var.get()),
            force_ocr=bool(self.force_ocr_var.get()),
            chapters_dir=chapters_dir,
        )

        self.run_btn.configure(state="disabled")
        self._set_status("Running...")

        def worker():
            try:
                chapters_text = pipeline.run(args)
                self.after(0, lambda: self._set_status(
                    f"Done! Output: {args.output}/final_output.mp4"))
                if chapters_text and self.show_chapters_popup_var.get():
                    self.after(0, lambda t=chapters_text: self._show_chapters(t))
                if self.close_on_complete_var.get():
                    self.after(0, self.destroy)
            except SystemExit as e:
                code = e.code
                self.after(0, lambda: self._set_status(f"Stopped (exit {code})"))
            except Exception as e:
                msg = str(e)
                self.after(0, lambda: self._set_status(f"Error: {msg}"))
            finally:
                self.after(0, lambda: self.run_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset(self):
        self.video_path = None
        self._thumb_ref = None
        self._thumb_pil = None
        self._img_rect = None
        self.thumb_canvas.delete("all")
        self._canvas_img_id = None
        self._canvas_rect_id = None
        self.info_label.configure(text="")
        default_out = os.path.expanduser(self.default_output_dir_var.get().strip())
        self.output_var.set(default_out if default_out else "out")
        self.t0_var.set("")
        self.chat_log_var.set("")
        self.verbose_var.set(False)
        self.preview_frame.pack_forget()
        self.run_btn.configure(state="disabled")
        self._set_status("Ready")

    # ------------------------------------------------------------------
    # Help window
    # ------------------------------------------------------------------

    def _show_help(self):
        import re
        readme_path = _resource_path("../`README.md")
        try:
            with open(readme_path, "r") as f:
                content = f.read()
        except OSError:
            content = "README.md not found."

        win = tk.Toplevel(self)
        win.title("Help")
        win.geometry("700x520")
        win.minsize(400, 300)

        style = ttk.Style()
        bg = style.lookup("TFrame", "background") or self.cget("bg")
        fg = style.lookup("TLabel", "foreground") or "black"
        code_bg = style.lookup("TEntry", "fieldbackground") or bg
        sel_bg = style.lookup("TEntry", "selectbackground") or "blue"
        sel_fg = style.lookup("TEntry", "selectforeground") or "white"

        font_name = style.lookup("TLabel", "font") or "TkDefaultFont"
        try:
            f = tkfont.Font(font=font_name)
            fam, fsz = f.actual("family"), f.actual("size")
        except Exception:
            fam, fsz = "TkDefaultFont", 10

        text = tk.Text(win, wrap=tk.WORD, padx=16, pady=12,
                       font=(fam, fsz), background=bg, foreground=fg,
                       relief=tk.FLAT, borderwidth=0, cursor="arrow",
                       selectbackground=sel_bg, selectforeground=sel_fg)
        scrollbar = ttk.Scrollbar(win, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        text.tag_configure("h1", font=(fam, fsz + 7, "bold"), spacing1=14, spacing3=4)
        text.tag_configure("h2", font=(fam, fsz + 3, "bold"), spacing1=10, spacing3=3)
        text.tag_configure("h3", font=(fam, fsz + 1, "bold"), spacing1=8, spacing3=2)
        text.tag_configure("code_block", font=("Courier", fsz - 1), background=code_bg,
                           lmargin1=20, lmargin2=20, spacing1=1, spacing3=1)
        text.tag_configure("inline_code", font=("Courier", fsz - 1), background=code_bg)
        text.tag_configure("bold", font=(fam, fsz, "bold"))
        text.tag_configure("bullet", lmargin1=16, lmargin2=28)
        text.tag_configure("th", font=(fam, fsz, "bold"))
        text.tag_configure("sep", foreground="#888888")

        _TABLE_SEP = re.compile(r'^\|[-| :]+\|$')
        in_code = False
        for line in content.splitlines():
            if line.startswith("```"):
                in_code = not in_code
                text.insert(tk.END, "\n")
                continue
            if in_code:
                text.insert(tk.END, line + "\n", "code_block")
                continue
            if line.startswith("### "):
                text.insert(tk.END, line[4:] + "\n", "h3")
            elif line.startswith("## "):
                text.insert(tk.END, line[3:] + "\n", "h2")
            elif line.startswith("# "):
                text.insert(tk.END, line[2:] + "\n", "h1")
            elif line.startswith("|"):
                if _TABLE_SEP.match(line):
                    text.insert(tk.END, "─" * 60 + "\n", "sep")
                else:
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    for j, cell in enumerate(cells):
                        if j:
                            text.insert(tk.END, "  │  ", "sep")
                        self._md_inline(text, cell, "th" if j == 0 else None)
                    text.insert(tk.END, "\n")
            elif re.match(r"^[-*+] ", line):
                text.insert(tk.END, "• ", "bullet")
                self._md_inline(text, line[2:], "bullet")
                text.insert(tk.END, "\n")
            elif line.strip() == "":
                text.insert(tk.END, "\n")
            else:
                self._md_inline(text, line)
                text.insert(tk.END, "\n")

        text.configure(state="disabled")

    def _show_chapters(self, chapters_text: str):
        """Open a window showing YouTube chapter timestamps with a copy button."""
        win = tk.Toplevel(self)
        win.title("YouTube Chapter Timestamps")
        win.resizable(True, True)
        win.minsize(300, 200)

        ttk.Label(
            win,
            text="Paste these into your YouTube video description:",
            padding=(12, 10, 12, 4),
        ).pack(anchor=tk.W)

        text_frame = ttk.Frame(win, padding=(12, 0, 12, 0))
        text_frame.pack(fill=tk.BOTH, expand=True)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.NONE,
            font=("Courier", 11),
            height=min(20, chapters_text.count("\n") + 2),
            width=30,
        )
        text_widget.insert(tk.END, chapters_text)
        text_widget.configure(state="disabled")
        text_widget.grid(row=0, column=0, sticky=tk.NSEW)

        sb = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky=tk.NS)

        btn_frame = ttk.Frame(win, padding=(12, 8))
        btn_frame.pack(fill=tk.X)

        def _copy():
            win.clipboard_clear()
            win.clipboard_append(chapters_text)
            copy_btn.configure(text="Copied!")
            win.after(1500, lambda: copy_btn.configure(text="Copy to Clipboard"))

        copy_btn = ttk.Button(btn_frame, text="Copy to Clipboard", command=_copy)
        copy_btn.pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        win.update_idletasks()
        win.geometry(f"{win.winfo_reqwidth() + 20}x{win.winfo_reqheight()}")

    def _md_inline(self, widget, text, base_tag=None):
        """Insert text with inline `code` and **bold** formatting applied."""
        import re
        for part in re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", text):
            if part.startswith("`") and part.endswith("`"):
                tags = ("inline_code", base_tag) if base_tag else ("inline_code",)
                widget.insert(tk.END, part[1:-1], tags)
            elif part.startswith("**") and part.endswith("**"):
                tags = ("bold", base_tag) if base_tag else ("bold",)
                widget.insert(tk.END, part[2:-2], tags)
            else:
                widget.insert(tk.END, part, (base_tag,) if base_tag else ())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_config(self, *_):
        cfg.save_config({
            "video_dir": self.default_video_dir_var.get(),
            "log_dir": self.default_log_dir_var.get(),
            "output_dir": self.default_output_dir_var.get(),
            "chapters_dir": self.default_chapters_dir_var.get(),
            "show_chapters_popup": bool(self.show_chapters_popup_var.get()),
            "close_on_complete": bool(self.close_on_complete_var.get()),
            "chat_region": self._chat_region,
            "threads": self.threads_var.get(),
            "ram_cap_gb": self.ram_cap_var.get(),
        })

    @staticmethod
    def _find_latest_video(video_dir: str) -> str | None:
        """Return the path of the most recently modified video file in video_dir."""
        candidates = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
            and os.path.isfile(os.path.join(video_dir, f))
        ]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    @staticmethod
    def _find_closest_log(video_path: str, log_dir: str) -> str | None:
        """Return the path of the 'Local' log file whose mtime is closest to the video's mtime."""
        video_mtime = os.path.getmtime(video_path)
        candidates = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if "Local" in f and os.path.isfile(os.path.join(log_dir, f))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(os.path.getmtime(p) - video_mtime))

    def _set_status(self, msg: str):
        self.status_label.configure(text=msg)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


if __name__ == "__main__":
    App().mainloop()
