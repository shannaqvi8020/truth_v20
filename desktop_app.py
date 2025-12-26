#!/usr/bin/env python3
"""
TruthScript - Local Transcription Desktop App
by Robert Michon

Production-ready Windows desktop application with native styling.
"""

import sys
import os

# CRITICAL FIX: PyInstaller windowed apps set stdout/stderr to None
# This causes crashes when any library tries to print or use tqdm
# Must be fixed BEFORE any other imports
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# Disable tqdm progress bars that can cause issues
os.environ['TQDM_DISABLE'] = '1'

if getattr(sys, 'frozen', False):
    bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import json
from pathlib import Path
from datetime import datetime

try:
    from transcribe import (
        load_whisper_model, 
        transcribe_audio, 
        is_model_downloaded, 
        get_device_info,
        get_supported_extensions,
        find_ffmpeg,
        is_video_file,
        setup_ffmpeg_path
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    MODULES_AVAILABLE = False
    find_ffmpeg = lambda: None
    is_video_file = lambda x: x.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
    setup_ffmpeg_path = lambda: None
    def get_supported_extensions():
        return ['.mp3', '.wav', '.m4a', '.aac', '.mp4', '.mov', '.flac', '.ogg', '.wma']
    def is_model_downloaded(name):
        return False
    def load_whisper_model(name, progress_callback=None):
        raise ImportError("Transcription modules not available")
    def transcribe_audio(*args, **kwargs):
        raise ImportError("Transcription modules not available")
    def get_device_info():
        return {'device': 'cpu', 'device_name': 'CPU', 'acceleration': False}


class TruthScriptApp:
    """Production-ready Windows application."""
    
    # Professional color scheme
    BG = '#f5f5f5'
    CARD = '#ffffff'
    BORDER = '#d0d0d0'
    PRIMARY = '#8b6914'
    PRIMARY_HOVER = '#6b5010'
    TEXT = '#1a1a1a'
    TEXT_MUTED = '#666666'
    SUCCESS = '#107c10'
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TruthScript - Transcription App")
        
        # Set DPI awareness for Windows
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
        
        # Large window to fit all controls comfortably
        self.root.geometry("1200x900")
        self.root.minsize(1100, 800)
        self.root.configure(bg=self.BG)
        
        # State
        self.model = None
        self.current_model_size = None
        self.selected_file = None
        self.save_folder = str(Path.home() / "Documents")
        self.result = None
        self.is_processing = False
        
        # Variables
        self.output_txt = tk.BooleanVar(value=True)
        self.output_md = tk.BooleanVar(value=True)
        self.output_json = tk.BooleanVar(value=False)
        self.model_choice = tk.StringVar(value="base")
        self.threshold = tk.DoubleVar(value=0.80)
        self.progress = tk.DoubleVar(value=0)
        
        self._setup_styles()
        self._build_ui()
    
    def _setup_styles(self):
        """Configure native Windows styling."""
        style = ttk.Style()
        
        # Use vista theme for native Windows look
        available_themes = style.theme_names()
        if 'vista' in available_themes:
            style.theme_use('vista')
        elif 'winnative' in available_themes:
            style.theme_use('winnative')
        else:
            style.theme_use('clam')
        
        # Configure widget styles
        style.configure('TCombobox', padding=8, font=('Segoe UI', 11))
        style.configure('TCheckbutton', font=('Segoe UI', 11))
        style.configure('TProgressbar', thickness=10)
        style.configure('TScale', sliderlength=20)
    
    def _build_ui(self):
        """Build the complete UI."""
        # Header bar
        self._build_header()
        
        # Main content with two columns
        main = tk.Frame(self.root, bg=self.BG)
        main.pack(fill='both', expand=True, padx=30, pady=20)
        
        # Left column - Settings (fixed width, no scrolling needed with large window)
        left_card = tk.Frame(main, bg=self.CARD, bd=1, relief='solid', width=420)
        left_card.pack(side='left', fill='y', padx=(0, 15))
        left_card.pack_propagate(False)
        
        self._build_settings_panel(left_card)
        
        # Right column - Preview (expands)
        right_card = tk.Frame(main, bg=self.CARD, bd=1, relief='solid')
        right_card.pack(side='left', fill='both', expand=True, padx=(15, 0))
        
        self._build_preview_panel(right_card)
    
    def _build_header(self):
        """Build header bar with branding."""
        header = tk.Frame(self.root, bg=self.CARD, height=70)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # Left - Title
        left = tk.Frame(header, bg=self.CARD)
        left.pack(side='left', padx=30, pady=15)
        
        title = tk.Label(left, text="TruthScript",
                        font=('Segoe UI Semibold', 24),
                        fg=self.PRIMARY, bg=self.CARD)
        title.pack(side='left')
        
        subtitle = tk.Label(left, text="by Robert Michon",
                           font=('Segoe UI', 11),
                           fg=self.TEXT_MUTED, bg=self.CARD)
        subtitle.pack(side='left', padx=(15, 0), pady=(10, 0))
        
        # Right - Status badges
        right = tk.Frame(header, bg=self.CARD)
        right.pack(side='right', padx=30, pady=15)
        
        # Device info
        device_info = get_device_info()
        if device_info.get('acceleration'):
            device_text = f"GPU: {device_info.get('device_name', 'GPU')}"
            device_color = self.SUCCESS
        else:
            device_text = "CPU Mode"
            device_color = self.PRIMARY
        
        tk.Label(right, text="100% Local  •  ",
                font=('Segoe UI', 11),
                fg=self.TEXT_MUTED, bg=self.CARD).pack(side='left')
        
        tk.Label(right, text=device_text,
                font=('Segoe UI Semibold', 11),
                fg=device_color, bg=self.CARD).pack(side='left')
        
        # Bottom border
        border = tk.Frame(self.root, bg=self.BORDER, height=1)
        border.pack(fill='x')
    
    def _build_settings_panel(self, parent):
        """Build settings panel with all controls."""
        # Scrollable content area
        content = tk.Frame(parent, bg=self.CARD)
        content.pack(fill='both', expand=True, padx=24, pady=24)
        
        # ===== FILE SELECTION =====
        self._section_label(content, "1. Select Audio/Video File")
        
        file_display = tk.Frame(content, bg='#f0f0f0', bd=1, relief='solid')
        file_display.pack(fill='x', pady=(12, 0))
        
        self.file_label = tk.Label(file_display,
                                   text="No file selected",
                                   font=('Segoe UI', 11),
                                   fg=self.TEXT_MUTED, bg='#f0f0f0',
                                   anchor='w', padx=15, pady=15)
        self.file_label.pack(fill='x')
        
        browse_btn = tk.Button(content, text="Browse Files...",
                              font=('Segoe UI', 11),
                              bg='white', fg=self.PRIMARY,
                              activebackground='#f0f0f0',
                              relief='solid', bd=1,
                              cursor='hand2', pady=10,
                              command=self._browse_file)
        browse_btn.pack(fill='x', pady=(12, 0))
        
        self._spacer(content, 24)
        
        # ===== MODEL SELECTION =====
        self._section_label(content, "2. Choose Whisper Model")
        
        model_frame = tk.Frame(content, bg=self.CARD)
        model_frame.pack(fill='x', pady=(12, 0))
        
        models = [
            ("tiny", "Tiny - Fastest"),
            ("base", "Base - Balanced"),
            ("small", "Small - Accurate"),
            ("medium", "Medium - More Accurate"),
            ("large", "Large - Most Accurate")
        ]
        
        model_combo = ttk.Combobox(model_frame,
                                   textvariable=self.model_choice,
                                   values=[m[0] for m in models],
                                   state='readonly',
                                   font=('Segoe UI', 11),
                                   width=35)
        model_combo.pack(fill='x')
        
        self._spacer(content, 24)
        
        # ===== CONFIDENCE THRESHOLD =====
        thresh_header = tk.Frame(content, bg=self.CARD)
        thresh_header.pack(fill='x')
        
        tk.Label(thresh_header, text="3. Confidence Threshold",
                font=('Segoe UI Semibold', 12),
                fg=self.TEXT, bg=self.CARD).pack(side='left')
        
        self.thresh_value_label = tk.Label(thresh_header, text="0.80",
                                           font=('Segoe UI Semibold', 12),
                                           fg=self.PRIMARY, bg=self.CARD)
        self.thresh_value_label.pack(side='right')
        
        thresh_scale = ttk.Scale(content, from_=0.50, to=0.95,
                                variable=self.threshold,
                                command=self._on_threshold_change)
        thresh_scale.pack(fill='x', pady=(12, 0))
        
        tk.Label(content, text="Words below this confidence will be marked [[UNKNOWN]]",
                font=('Segoe UI', 10),
                fg=self.TEXT_MUTED, bg=self.CARD).pack(anchor='w', pady=(6, 0))
        
        self._spacer(content, 24)
        
        # ===== OUTPUT FORMATS =====
        self._section_label(content, "4. Output Formats")
        
        formats_frame = tk.Frame(content, bg=self.CARD)
        formats_frame.pack(fill='x', pady=(12, 0))
        
        ttk.Checkbutton(formats_frame, text="Plain Text (.txt)",
                       variable=self.output_txt).pack(anchor='w', pady=4)
        ttk.Checkbutton(formats_frame, text="Markdown (.md)",
                       variable=self.output_md).pack(anchor='w', pady=4)
        ttk.Checkbutton(formats_frame, text="JSON with timestamps",
                       variable=self.output_json).pack(anchor='w', pady=4)
        
        self._spacer(content, 24)
        
        # ===== SAVE LOCATION =====
        self._section_label(content, "5. Save Location")
        
        save_row = tk.Frame(content, bg=self.CARD)
        save_row.pack(fill='x', pady=(12, 0))
        
        self.save_label = tk.Label(save_row,
                                   text=self._truncate_path(self.save_folder),
                                   font=('Segoe UI', 10),
                                   fg=self.TEXT_MUTED, bg='#f0f0f0',
                                   anchor='w', padx=12, pady=12)
        self.save_label.pack(side='left', fill='x', expand=True)
        
        choose_btn = tk.Button(save_row, text="Change...",
                              font=('Segoe UI', 10),
                              bg='white', fg=self.TEXT_MUTED,
                              relief='solid', bd=1,
                              cursor='hand2', padx=15, pady=6,
                              command=self._choose_folder)
        choose_btn.pack(side='right', padx=(12, 0))
        
        # ===== TRANSCRIBE BUTTON (at bottom) =====
        spacer = tk.Frame(content, bg=self.CARD)
        spacer.pack(fill='both', expand=True)
        
        self._spacer(content, 20)
        
        self.transcribe_btn = tk.Button(content, text="TRANSCRIBE",
                                        font=('Segoe UI Semibold', 16),
                                        bg=self.PRIMARY, fg='white',
                                        activebackground=self.PRIMARY_HOVER,
                                        activeforeground='white',
                                        relief='flat', cursor='hand2',
                                        pady=16,
                                        command=self._start_transcription)
        self.transcribe_btn.pack(fill='x')
    
    def _build_preview_panel(self, parent):
        """Build transcript preview panel."""
        content = tk.Frame(parent, bg=self.CARD)
        content.pack(fill='both', expand=True, padx=24, pady=24)
        
        # Header
        tk.Label(content, text="Transcript Preview",
                font=('Segoe UI Semibold', 16),
                fg=self.TEXT, bg=self.CARD).pack(anchor='w')
        
        # Stats bar
        stats_frame = tk.Frame(content, bg='#f0f0f0')
        stats_frame.pack(fill='x', pady=(16, 0))
        
        self.stats_label = tk.Label(stats_frame,
                                    text="Words: —   •   Unknown: —   •   Critical: —   •   Accuracy: —",
                                    font=('Segoe UI', 11),
                                    fg=self.TEXT_MUTED, bg='#f0f0f0',
                                    pady=12)
        self.stats_label.pack()
        
        # Progress bar
        progress_frame = tk.Frame(content, bg=self.CARD)
        progress_frame.pack(fill='x', pady=(20, 0))
        
        self.progress_bar = ttk.Progressbar(progress_frame,
                                            variable=self.progress,
                                            maximum=100,
                                            mode='determinate')
        self.progress_bar.pack(fill='x')
        
        self.progress_label = tk.Label(progress_frame, text="Ready to transcribe",
                                       font=('Segoe UI', 10),
                                       fg=self.TEXT_MUTED, bg=self.CARD)
        self.progress_label.pack(anchor='w', pady=(8, 0))
        
        # Text preview with scrollbar
        text_container = tk.Frame(content, bg='white', bd=1, relief='solid')
        text_container.pack(fill='both', expand=True, pady=(20, 0))
        
        scrollbar = ttk.Scrollbar(text_container)
        scrollbar.pack(side='right', fill='y')
        
        self.preview_text = tk.Text(text_container,
                                    font=('Consolas', 11),
                                    bg='white', fg=self.TEXT,
                                    wrap='word', relief='flat',
                                    padx=20, pady=20,
                                    yscrollcommand=scrollbar.set)
        self.preview_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.preview_text.yview)
        
        self.preview_text.insert('1.0', 
            "Your transcript will appear here.\n\n"
            "Steps:\n"
            "1. Select an audio or video file\n"
            "2. Choose your settings\n"
            "3. Click TRANSCRIBE\n\n"
            "Low-confidence words will be marked as [[UNKNOWN]].")
        self.preview_text.config(state='disabled')
        
        # Completion message
        self.complete_label = tk.Label(content, text="",
                                       font=('Segoe UI', 11),
                                       fg=self.SUCCESS, bg=self.CARD)
        self.complete_label.pack(anchor='w', pady=(16, 0))
    
    def _section_label(self, parent, text):
        """Create a section label."""
        tk.Label(parent, text=text,
                font=('Segoe UI Semibold', 12),
                fg=self.TEXT, bg=self.CARD).pack(anchor='w')
    
    def _spacer(self, parent, height):
        """Create vertical spacing."""
        tk.Frame(parent, bg=self.CARD, height=height).pack(fill='x')
    
    def _truncate_path(self, path, max_len=40):
        """Truncate path for display."""
        if len(path) <= max_len:
            return path
        return "..." + path[-(max_len-3):]
    
    def _on_threshold_change(self, value):
        """Update threshold display."""
        self.thresh_value_label.config(text=f"{float(value):.2f}")
    
    def _browse_file(self):
        """Browse for audio/video file."""
        extensions = get_supported_extensions()
        filetypes = [
            ("Audio/Video Files", " ".join(f"*{e}" for e in extensions)),
            ("All Files", "*.*")
        ]
        
        path = filedialog.askopenfilename(title="Select Audio or Video File",
                                          filetypes=filetypes)
        if path:
            self.selected_file = path
            name = os.path.basename(path)
            size = os.path.getsize(path) / (1024 * 1024)
            self.file_label.config(text=f"{name}  ({size:.1f} MB)",
                                  fg=self.TEXT)
    
    def _choose_folder(self):
        """Choose save folder."""
        folder = filedialog.askdirectory(title="Choose Save Location",
                                        initialdir=self.save_folder)
        if folder:
            self.save_folder = folder
            self.save_label.config(text=self._truncate_path(folder))
    
    def _start_transcription(self):
        """Start transcription process."""
        if not self.selected_file:
            messagebox.showwarning("No File Selected", 
                                   "Please select an audio or video file first.")
            return
        
        if not MODULES_AVAILABLE:
            messagebox.showerror("Error", "Transcription modules not available.")
            return
        
        if not (self.output_txt.get() or self.output_md.get() or self.output_json.get()):
            messagebox.showwarning("No Format Selected",
                                   "Please select at least one output format.")
            return
        
        # Check for FFmpeg if this is a video file
        if is_video_file(self.selected_file):
            setup_ffmpeg_path()  # Add bundled ffmpeg to PATH
            import shutil
            ffmpeg_path = find_ffmpeg()
            if not ffmpeg_path:
                ffmpeg_path = shutil.which('ffmpeg')
            
            if not ffmpeg_path:
                # Build helpful error message
                exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
                messagebox.showerror(
                    "FFmpeg Required",
                    f"Video files require FFmpeg to extract audio.\n\n"
                    f"Please download FFmpeg from:\n"
                    f"https://www.gyan.dev/ffmpeg/builds/\n\n"
                    f"Then place ffmpeg.exe in:\n"
                    f"{os.path.join(exe_dir, 'ffmpeg_bin')}\n\n"
                    f"Or install FFmpeg and add it to your system PATH.\n\n"
                    f"Alternatively, convert your video to MP3/WAV first."
                )
                return
        
        self.is_processing = True
        self.transcribe_btn.config(state='disabled', text="Processing...",
                                   bg='#999999')
        self.progress.set(0)
        self.progress_label.config(text="Starting transcription...")
        self.complete_label.config(text="")
        
        thread = threading.Thread(target=self._run_transcription, daemon=True)
        thread.start()
    
    def _update_progress(self, value, message):
        """Thread-safe progress update."""
        self.root.after(0, lambda v=value, m=message: self._apply_progress(v, m))
    
    def _apply_progress(self, value, message):
        """Apply progress update to UI."""
        self.progress.set(min(value * 100, 100))
        self.progress_label.config(text=message)
        # Force UI update for smooth progress bar movement
        try:
            self.root.update_idletasks()
        except:
            pass
    
    def _run_transcription(self):
        """Run transcription in background thread."""
        try:
            model_name = self.model_choice.get()
            threshold = self.threshold.get()
            
            self._update_progress(0.02, f"Initializing {model_name} model...")
            
            try:
                if self.model is None or self.current_model_size != model_name:
                    if not is_model_downloaded(model_name):
                        self._update_progress(0.03, f"Downloading {model_name} model (first time only)...")
                    self.model = load_whisper_model(model_name, progress_callback=self._update_progress)
                    self.current_model_size = model_name
                else:
                    self._update_progress(0.65, f"Model {model_name} ready")
            except Exception as model_err:
                raise RuntimeError(f"Failed to load model: {model_err}")
            
            if self.model is None:
                raise RuntimeError("Model failed to load (returned None)")
            
            try:
                result = transcribe_audio(
                    audio_path=self.selected_file,
                    model=self.model,
                    confidence_threshold=threshold,
                    low_confidence_threshold=0.50,
                    progress_callback=self._update_progress
                )
            except Exception as trans_err:
                raise RuntimeError(f"Transcription failed: {trans_err}")
            
            if result is None:
                raise RuntimeError("Transcription returned None")
            
            self.result = result
            
            if result.get('success'):
                self._update_progress(0.97, "Saving output files...")
                try:
                    self._save_outputs(result)
                except Exception as save_err:
                    raise RuntimeError(f"Failed to save outputs: {save_err}")
                self._update_progress(1.0, "Complete!")
            
            self.root.after(0, lambda: self._show_results(result))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._show_error(str(e)))
    
    def _save_outputs(self, result):
        """Save transcript files."""
        base_name = os.path.splitext(os.path.basename(self.selected_file))[0]
        folder = self.save_folder
        os.makedirs(folder, exist_ok=True)
        
        total = result.get('total_words', 0)
        unknown = result.get('unknown_count', 0)
        critical = result.get('unknown_star_count', 0)
        transcript = result.get('cleaned_text', '')
        
        if self.output_txt.get():
            path = os.path.join(folder, f"{base_name}_transcript.txt")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(transcript)
        
        if self.output_md.get():
            rate = (unknown / total * 100) if total > 0 else 0
            md_content = f"# Transcript: {base_name}\n\n"
            md_content += f"**Words:** {total} | **Unknown:** {unknown} | **Critical:** {critical} | **Accuracy:** {100-rate:.1f}%\n\n"
            md_content += "---\n\n"
            md_content += transcript
            path = os.path.join(folder, f"{base_name}_transcript.md")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        if self.output_json.get():
            json_data = {
                "metadata": {
                    "source_file": os.path.basename(self.selected_file),
                    "model": self.model_choice.get(),
                    "confidence_threshold": self.threshold.get(),
                    "total_words": total,
                    "unknown_count": unknown,
                    "critical_count": critical,
                    "created_at": datetime.now().isoformat()
                },
                "words": result.get('words', []),
                "transcript": transcript
            }
            path = os.path.join(folder, f"{base_name}_transcript.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _show_results(self, result):
        """Display transcription results."""
        self.is_processing = False
        self.transcribe_btn.config(state='normal', text="TRANSCRIBE",
                                   bg=self.PRIMARY)
        
        if not result.get('success'):
            self._show_error(result.get('error', 'Unknown error occurred'))
            return
        
        total = result.get('total_words', 0)
        unknown = result.get('unknown_count', 0)
        critical = result.get('unknown_star_count', 0)
        accuracy = (1 - unknown / total) * 100 if total > 0 else 100
        
        self.stats_label.config(
            text=f"Words: {total}   •   Unknown: {unknown}   •   Critical: {critical}   •   Accuracy: {accuracy:.1f}%"
        )
        
        self.preview_text.config(state='normal')
        self.preview_text.delete('1.0', 'end')
        self.preview_text.insert('1.0', result.get('cleaned_text', ''))
        self.preview_text.config(state='disabled')
        
        formats_saved = []
        if self.output_txt.get(): formats_saved.append(".txt")
        if self.output_md.get(): formats_saved.append(".md")
        if self.output_json.get(): formats_saved.append(".json")
        
        self.complete_label.config(
            text=f"✓ Saved {', '.join(formats_saved)} files to: {self.save_folder}"
        )
    
    def _show_error(self, message):
        """Show error state."""
        self.is_processing = False
        self.transcribe_btn.config(state='normal', text="TRANSCRIBE",
                                   bg=self.PRIMARY)
        self.progress_label.config(text="Error occurred")
        self.progress.set(0)
        messagebox.showerror("Transcription Error", message)
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Entry point."""
    app = TruthScriptApp()
    app.run()


if __name__ == "__main__":
    main()
