#!/usr/bin/env python3
"""
TruthScript - Transcription Engine
Core transcription logic using OpenAI Whisper with confidence thresholding.
"""

import os
import sys
import io
import tempfile
import hashlib
import urllib.request
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Callable


# CRITICAL FIX: PyInstaller windowed apps set stdout/stderr to None
# This causes tqdm (used by Whisper) to crash with "'NoneType' object has no attribute 'write'"
# Fix by redirecting to devnull or a dummy stream
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# Also disable tqdm progress bars to prevent any issues
os.environ['TQDM_DISABLE'] = '1'


def setup_ffmpeg_path():
    """
    Ensure FFmpeg is available on PATH before any audio operations.
    This is critical for the packaged .exe to work.
    """
    # Check if ffmpeg already works
    if shutil.which('ffmpeg'):
        return True
    
    # Possible FFmpeg locations for packaged app
    search_paths = []
    
    if getattr(sys, 'frozen', False):
        # Running as packaged .exe
        exe_dir = os.path.dirname(sys.executable)
        bundle_dir = getattr(sys, '_MEIPASS', exe_dir)
        
        search_paths = [
            os.path.join(bundle_dir, 'ffmpeg_bin'),
            os.path.join(bundle_dir, '_internal', 'ffmpeg_bin'),
            os.path.join(exe_dir, 'ffmpeg_bin'),
            os.path.join(exe_dir, '_internal', 'ffmpeg_bin'),
            bundle_dir,
            exe_dir,
        ]
    else:
        # Running from source
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths = [
            os.path.join(script_dir, 'ffmpeg_bin'),
        ]
    
    # Search for ffmpeg and add to PATH
    for path in search_paths:
        if os.path.isdir(path):
            ffmpeg_exe = os.path.join(path, 'ffmpeg.exe')
            if os.path.isfile(ffmpeg_exe):
                current_path = os.environ.get('PATH', '')
                if path not in current_path:
                    os.environ['PATH'] = path + os.pathsep + current_path
                return True
    
    return False


def get_ffmpeg_path() -> Optional[str]:
    """Get the path to ffmpeg executable."""
    setup_ffmpeg_path()
    
    # Try to find ffmpeg
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    
    # Manual search
    search_paths = []
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        bundle_dir = getattr(sys, '_MEIPASS', exe_dir)
        search_paths = [
            os.path.join(bundle_dir, 'ffmpeg_bin', 'ffmpeg.exe'),
            os.path.join(bundle_dir, '_internal', 'ffmpeg_bin', 'ffmpeg.exe'),
            os.path.join(exe_dir, 'ffmpeg_bin', 'ffmpeg.exe'),
            os.path.join(exe_dir, '_internal', 'ffmpeg_bin', 'ffmpeg.exe'),
        ]
    
    for path in search_paths:
        if os.path.isfile(path):
            return path
    
    return None


# Initialize FFmpeg path at module load
_ffmpeg_initialized = setup_ffmpeg_path()


import torch
import whisper


# Speed optimization: Set PyTorch to use optimized backends
torch.set_num_threads(os.cpu_count() or 4)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('medium')


# Whisper model download URLs and checksums
_MODELS = {
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

_MODEL_SIZES = {
    "tiny": 75,
    "base": 142,
    "small": 466,
    "medium": 1500,
    "large": 3000,
}


def get_device() -> str:
    """
    Detect the best available device for PyTorch.
    Returns: 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' as fallback
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """Get detailed information about the compute device."""
    device = get_device()
    info = {
        'device': device,
        'device_name': 'CPU',
        'acceleration': False,
        'memory_gb': None
    }
    
    if device == 'cuda':
        info['device_name'] = torch.cuda.get_device_name(0)
        info['acceleration'] = True
        info['cuda_version'] = torch.version.cuda
        info['memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    elif device == 'mps':
        info['device_name'] = 'Apple Silicon GPU (MPS)'
        info['acceleration'] = True
    
    return info


def get_model_path() -> str:
    """Get the path where Whisper models are stored."""
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), 'models')
    return os.path.join(os.path.expanduser('~'), '.cache', 'whisper')


def is_model_downloaded(model_name: str = "base") -> bool:
    """Check if a Whisper model is already downloaded."""
    model_path = get_model_path()
    
    # Handle special naming for large model
    if model_name == "large":
        model_file = os.path.join(model_path, "large-v3.pt")
    else:
        model_file = os.path.join(model_path, f"{model_name}.pt")
    
    return os.path.exists(model_file)


def download_model_with_progress(
    model_name: str,
    download_root: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> str:
    """Download Whisper model with real progress tracking."""
    
    os.makedirs(download_root, exist_ok=True)
    
    # Handle large model naming
    if model_name == "large":
        local_file = os.path.join(download_root, "large-v3.pt")
    else:
        local_file = os.path.join(download_root, f"{model_name}.pt")
    
    # If already exists, skip download
    if os.path.exists(local_file):
        if progress_callback:
            progress_callback(0.10, f"Model {model_name} already downloaded")
        return local_file
    
    url = _MODELS.get(model_name)
    if not url:
        raise ValueError(f"Unknown model: {model_name}")
    
    size_mb = _MODEL_SIZES.get(model_name, 100)
    
    if progress_callback:
        progress_callback(0.01, f"Starting download of {model_name} model ({size_mb} MB)...")
    
    # Download with progress
    temp_file = local_file + ".download"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                total_size = size_mb * 1024 * 1024
            
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            
            with open(temp_file, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback:
                        pct = min(downloaded / total_size, 1.0)
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        # Map download progress to 0.01 - 0.50 range (50% of bar for download)
                        progress_val = 0.01 + (pct * 0.49)
                        progress_callback(
                            progress_val, 
                            f"Downloading {model_name}: {downloaded_mb:.0f} / {total_mb:.0f} MB ({pct*100:.0f}%)"
                        )
        
        # Move temp file to final location
        os.rename(temp_file, local_file)
        
        if progress_callback:
            progress_callback(0.52, f"Download complete! Loading model...")
        
        return local_file
        
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise RuntimeError(f"Failed to download model: {str(e)}")


def load_whisper_model(
    model_name: str = "base",
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> whisper.Whisper:
    """Load Whisper model with progress callback for download and loading."""
    device = get_device()
    device_info = get_device_info()
    download_root = get_model_path()
    
    # Download model with progress if needed
    if not is_model_downloaded(model_name):
        download_model_with_progress(model_name, download_root, progress_callback)
        if progress_callback:
            progress_callback(0.55, f"Loading {model_name} model into memory...")
    else:
        if progress_callback:
            progress_callback(0.10, f"Loading {model_name} model...")
    
    if progress_callback:
        progress_callback(0.58, f"Loading model into {device_info['device_name']}...")
    
    # Load model
    model = whisper.load_model(model_name, device=device, download_root=download_root)
    
    if model is None:
        raise RuntimeError(f"Failed to load model {model_name}")
    
    # Speed optimization: Use half precision on GPU
    if device == 'cuda':
        model = model.half()
    
    if progress_callback:
        accel = "GPU accelerated" if device_info['acceleration'] else "CPU mode"
        progress_callback(0.65, f"Model ready ({accel})")
    
    return model


def is_video_file(filepath: str) -> bool:
    """Check if the file is a video format that needs audio extraction."""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}
    _, ext = os.path.splitext(filepath.lower())
    return ext in video_extensions


def find_ffmpeg() -> str:
    """Find ffmpeg executable, checking multiple locations."""
    import shutil
    
    # List of places to check for ffmpeg
    candidates = []
    
    if getattr(sys, 'frozen', False):
        # Running as .exe - check bundled locations
        exe_dir = os.path.dirname(sys.executable)
        meipass = getattr(sys, '_MEIPASS', exe_dir)
        
        # Check various bundled locations (most likely first)
        candidates.extend([
            os.path.join(exe_dir, 'ffmpeg_bin', 'ffmpeg.exe'),
            os.path.join(exe_dir, '_internal', 'ffmpeg_bin', 'ffmpeg.exe'),
            os.path.join(meipass, 'ffmpeg_bin', 'ffmpeg.exe'),
            os.path.join(meipass, 'ffmpeg.exe'),
            os.path.join(exe_dir, 'ffmpeg.exe'),
        ])
    else:
        # Development mode
        candidates.append(os.path.join(os.path.dirname(__file__), 'ffmpeg_bin', 'ffmpeg.exe'))
    
    # Check each candidate
    for path in candidates:
        if os.path.isfile(path):
            return path
    
    # Fall back to system PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    return None


def get_ffmpeg_debug_info() -> str:
    """Get debug info about FFmpeg search locations."""
    import shutil
    
    lines = []
    
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        meipass = getattr(sys, '_MEIPASS', exe_dir)
        
        lines.append(f"exe_dir: {exe_dir}")
        lines.append(f"_MEIPASS: {meipass}")
        
        # Check what folders exist
        for folder in ['ffmpeg_bin', '_internal', '_internal/ffmpeg_bin']:
            full_path = os.path.join(exe_dir, folder)
            exists = os.path.exists(full_path)
            lines.append(f"  {folder}: {'EXISTS' if exists else 'MISSING'}")
            if exists and os.path.isdir(full_path):
                try:
                    contents = os.listdir(full_path)[:5]
                    lines.append(f"    contents: {contents}")
                except:
                    pass
    
    ffmpeg = find_ffmpeg()
    lines.append(f"find_ffmpeg result: {ffmpeg}")
    
    system_ffmpeg = shutil.which('ffmpeg')
    lines.append(f"system PATH ffmpeg: {system_ffmpeg}")
    
    return "\n".join(lines)


def setup_ffmpeg_path():
    """Add bundled ffmpeg to PATH if running as frozen exe."""
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        
        # Add all possible ffmpeg locations to PATH
        ffmpeg_dirs = [
            os.path.join(exe_dir, 'ffmpeg_bin'),
            os.path.join(exe_dir, '_internal', 'ffmpeg_bin'),
            exe_dir,
        ]
        
        current_path = os.environ.get('PATH', '')
        for ffmpeg_dir in ffmpeg_dirs:
            if os.path.isdir(ffmpeg_dir) and ffmpeg_dir not in current_path:
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
                current_path = os.environ['PATH']


def extract_audio_with_ffmpeg(input_path: str, output_path: str) -> str:
    """Extract audio from video file using ffmpeg subprocess."""
    import subprocess
    import shutil
    
    # Ensure ffmpeg is on PATH
    setup_ffmpeg_path()
    
    # Convert to absolute paths
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    # Try to find ffmpeg - first check our bundled locations
    ffmpeg_cmd = find_ffmpeg()
    
    # If not found, try shutil.which (checks PATH)
    if not ffmpeg_cmd:
        ffmpeg_cmd = shutil.which('ffmpeg')
    
    # If still not found, try just 'ffmpeg' and let Windows find it
    if not ffmpeg_cmd:
        ffmpeg_cmd = 'ffmpeg'
    
    # Verify input file exists
    if not os.path.isfile(input_path):
        raise RuntimeError(f"Input file not found: {input_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use simple command list - let Python handle escaping
        cmd = [
            ffmpeg_cmd,
            '-i', input_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]
        
        # Run with subprocess.run, shell=False for security
        # No timeout - allow long files to complete
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(input_path) or None  # Set working directory to input file location
        )
        
        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg error: {error_msg}")
        
        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg did not produce output file")
            
        return output_path
        
    except FileNotFoundError as e:
        # FFmpeg not found - give helpful error
        debug_info = get_ffmpeg_debug_info()
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            raise RuntimeError(
                f"FFmpeg not found!\n\n"
                f"Please download FFmpeg and place ffmpeg.exe in:\n"
                f"{os.path.join(exe_dir, 'ffmpeg_bin')}\n\n"
                f"Or install FFmpeg and add it to your system PATH.\n\n"
                f"Debug info:\n{debug_info}"
            )
        else:
            raise RuntimeError(f"FFmpeg not found: {e}\n\nDebug:\n{debug_info}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out (10 min limit).")
    except OSError as e:
        error_code = getattr(e, 'winerror', getattr(e, 'errno', 'N/A'))
        debug_info = get_ffmpeg_debug_info()
        raise RuntimeError(
            f"OS Error (code {error_code}): {e}\n\n"
            f"Debug info:\n{debug_info}"
        )


def transcribe_audio(
    audio_path: str,
    model: whisper.Whisper,
    confidence_threshold: float = 0.80,
    low_confidence_threshold: float = 0.50,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper with word-level timestamps.
    """
    if progress_callback:
        progress_callback(0.66, "Preparing audio file...")
    
    temp_audio_path = None
    audio_to_transcribe = audio_path
    copied_temp_file = None
    wav_temp_path = None
    
    try:
        # Normalize the path and handle special characters
        audio_path = os.path.normpath(audio_path)
        
        # Check if file exists first
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio file not found: {audio_path}")
        
        # If path has non-ASCII chars or is very long, copy to temp with simple name
        try:
            audio_path.encode('ascii')
            path_ok = len(audio_path) < 200
        except UnicodeEncodeError:
            path_ok = False
        
        if not path_ok:
            if progress_callback:
                progress_callback(0.67, "Copying file to temp location...")
            # Copy to temp with simple ASCII name
            ext = os.path.splitext(audio_path)[1].lower()
            temp_fd, copied_temp_file = tempfile.mkstemp(suffix=ext)
            os.close(temp_fd)
            shutil.copy2(audio_path, copied_temp_file)
            audio_path = copied_temp_file
        
        audio_to_transcribe = audio_path
        
        # Extract audio from video if needed
        if is_video_file(audio_path):
            if progress_callback:
                progress_callback(0.68, "Extracting audio from video...")
            
            temp_fd, temp_audio_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            audio_to_transcribe = extract_audio_with_ffmpeg(audio_path, temp_audio_path)
            
            # Validate extraction result
            if audio_to_transcribe is None:
                raise RuntimeError("Failed to extract audio from video - FFmpeg returned no output")
            if not os.path.exists(audio_to_transcribe):
                raise RuntimeError(f"Audio extraction failed - output file not created: {audio_to_transcribe}")
            if os.path.getsize(audio_to_transcribe) == 0:
                raise RuntimeError("Audio extraction failed - output file is empty")
        
        # Validate audio file before transcription
        if not audio_to_transcribe or not os.path.exists(audio_to_transcribe):
            raise RuntimeError(f"Audio file not found: {audio_to_transcribe}")
        
        file_size = os.path.getsize(audio_to_transcribe)
        if file_size == 0:
            raise RuntimeError("Audio file is empty (0 bytes)")
        if file_size < 100:
            raise RuntimeError(f"Audio file too small ({file_size} bytes) - likely corrupted")
        
        # Ensure FFmpeg is set up before loading audio
        setup_ffmpeg_path()
        
        if progress_callback:
            progress_callback(0.70, "Loading audio file...")
        
        # Convert audio to WAV format first for reliable processing
        temp_fd, wav_temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        
        try:
            ffmpeg_exe = get_ffmpeg_path() or 'ffmpeg'
            cmd = [
                ffmpeg_exe, '-y', '-i', audio_to_transcribe,
                '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
                wav_temp_path
            ]
            # No timeout - allow long files to complete
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"FFmpeg conversion failed: {stderr}")
            if not os.path.exists(wav_temp_path) or os.path.getsize(wav_temp_path) < 100:
                raise RuntimeError("Audio conversion produced empty or invalid file")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found!\n\n"
                "Please download FFmpeg from https://ffmpeg.org and:\n"
                "1. Place ffmpeg.exe in a 'ffmpeg_bin' folder next to TruthScript.exe\n"
                "2. Or add FFmpeg to your system PATH"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio conversion timed out. The file may be too large or corrupted.")
        
        # Use the converted WAV file
        audio_to_transcribe = wav_temp_path
        
        # Estimate duration based on file size (rough estimate: 16000 samples/sec * 2 bytes = 32KB/sec)
        wav_size = os.path.getsize(wav_temp_path)
        audio_duration_sec = wav_size / 32000
        audio_minutes = max(1, int(audio_duration_sec / 60))
        
        device = get_device()
        if device == 'cuda':
            est_time = max(1, audio_minutes // 3)
            time_msg = f"~{est_time} min with GPU"
        else:
            est_time = max(2, audio_minutes * 2)
            time_msg = f"~{est_time}-{est_time*2} min on CPU"
        
        if progress_callback:
            progress_callback(0.72, f"Transcribing {audio_minutes}min audio ({time_msg})...")
        
        # Transcribe with Whisper in a background thread so UI stays responsive
        import threading
        import time
        
        transcription_result = {"data": None, "error": None, "done": False}
        
        def run_transcription():
            try:
                transcription_result["data"] = model.transcribe(
                    audio_to_transcribe,
                    word_timestamps=True,
                    verbose=False,
                    fp16=(device == 'cuda'),
                    temperature=0,
                    beam_size=5,
                    best_of=5,
                    condition_on_previous_text=True,
                )
            except Exception as e:
                transcription_result["error"] = e
            finally:
                transcription_result["done"] = True
        
        # Start transcription in background thread
        transcribe_thread = threading.Thread(target=run_transcription, daemon=True)
        transcribe_thread.start()
        
        # Show progress updates while waiting
        progress_value = 0.72
        progress_step = 0.12 / max(1, est_time * 6)
        start_time = time.time()
        
        while not transcription_result["done"]:
            time.sleep(5)  # Check every 5 seconds
            elapsed_sec = time.time() - start_time
            elapsed_min = int(elapsed_sec / 60)
            
            if progress_value < 0.84:
                progress_value += progress_step
            
            if progress_callback:
                progress_callback(progress_value, f"Transcribing... {elapsed_min}min elapsed")
        
        # Check for errors
        if transcription_result["error"] is not None:
            err = transcription_result["error"]
            error_str = str(err)
            if "'NoneType'" in error_str and "write" in error_str:
                raise RuntimeError(
                    f"Audio processing failed. This can happen if:\n"
                    f"1. The audio file is corrupted\n"
                    f"2. The audio format is not supported\n\n"
                    f"Try converting your audio to MP3 or WAV format first."
                )
            raise RuntimeError(f"Transcription failed: {err}")
        
        result = transcription_result["data"]
        
        if result is None:
            raise RuntimeError("Transcription returned no result")
        
        if progress_callback:
            progress_callback(0.85, "Processing word-level confidence data...")
        
        # Extract word-level data
        words_data = []
        segments = result.get('segments', [])
        
        for segment in segments:
            for word_info in segment.get('words', []):
                word_entry = {
                    'word': word_info.get('word', '').strip(),
                    'start': round(word_info.get('start', 0), 3),
                    'end': round(word_info.get('end', 0), 3),
                    'confidence': round(word_info.get('probability', 1.0), 4),
                }
                words_data.append(word_entry)
        
        if progress_callback:
            progress_callback(0.90, "Applying confidence thresholds...")
        
        # Apply confidence thresholds
        cleaned_words = []
        unknown_count = 0
        unknown_star_count = 0
        
        for word_entry in words_data:
            word = word_entry['word']
            confidence = word_entry['confidence']
            
            if confidence < low_confidence_threshold:
                cleaned_words.append('[[UNKNOWN*]]')
                unknown_star_count += 1
                unknown_count += 1
            elif confidence < confidence_threshold:
                cleaned_words.append('[[UNKNOWN]]')
                unknown_count += 1
            else:
                cleaned_words.append(word)
        
        # Clean up text
        cleaned_text = ' '.join(cleaned_words)
        cleaned_text = cleaned_text.replace(' ,', ',').replace(' .', '.').replace(' ?', '?').replace(' !', '!')
        cleaned_text = cleaned_text.replace("' ", "'").replace(" '", "'")
        cleaned_text = cleaned_text.replace('  ', ' ').strip()
        
        if progress_callback:
            progress_callback(0.95, "Transcription complete!")
        
        return {
            'success': True,
            'cleaned_text': cleaned_text,
            'total_words': len(words_data),
            'unknown_count': unknown_count,
            'unknown_star_count': unknown_star_count,
            'words': words_data,
            'segments': segments,
            'language': result.get('language', 'unknown'),
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Transcription error: {error_details}")
        return {
            'success': False,
            'error': str(e),
            'cleaned_text': '',
            'total_words': 0,
            'unknown_count': 0,
            'unknown_star_count': 0,
            'words': [],
            'segments': [],
        }
    finally:
        # Clean up all temp files
        for temp_file in [temp_audio_path, copied_temp_file, wav_temp_path]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass


def get_model_info() -> Dict[str, str]:
    """Get information about available Whisper models."""
    return {
        'tiny': 'Fastest, lowest accuracy (~75 MB download)',
        'base': 'Good balance (~142 MB download)',
        'small': 'Better accuracy (~466 MB download)',
        'medium': 'High accuracy (~1.5 GB download)',
        'large': 'Best accuracy (~3 GB download)'
    }


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return ['.mp3', '.wav', '.m4a', '.aac', '.mp4', '.mov', '.flac', '.ogg', '.wma']
