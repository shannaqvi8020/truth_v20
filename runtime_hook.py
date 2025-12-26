"""
PyInstaller runtime hook for TruthScript - Windows
Sets up paths for bundled ffmpeg and Tcl/Tk libraries at the earliest possible point.
"""
import sys
import os

def setup_environment():
    """Set up all required environment variables and paths."""
    if not getattr(sys, 'frozen', False):
        return
    
    # Get the bundle directory (PyInstaller temp folder) and exe directory
    bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    exe_dir = os.path.dirname(sys.executable)
    
    # FFmpeg search paths - check multiple locations
    ffmpeg_locations = [
        os.path.join(exe_dir, 'ffmpeg_bin'),
        os.path.join(exe_dir, '_internal', 'ffmpeg_bin'),
        os.path.join(bundle_dir, 'ffmpeg_bin'),
        os.path.join(bundle_dir, '_internal', 'ffmpeg_bin'),
        exe_dir,
        bundle_dir,
    ]
    
    current_path = os.environ.get('PATH', '')
    paths_to_add = []
    
    for ffmpeg_dir in ffmpeg_locations:
        if os.path.isdir(ffmpeg_dir):
            ffmpeg_exe = os.path.join(ffmpeg_dir, 'ffmpeg.exe')
            if os.path.isfile(ffmpeg_exe):
                if ffmpeg_dir not in current_path:
                    paths_to_add.append(ffmpeg_dir)
    
    if paths_to_add:
        os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + current_path
    
    # Set Tcl/Tk library paths (for tkinter)
    tcl_dirs = [
        os.path.join(bundle_dir, 'tcl'),
        os.path.join(bundle_dir, 'tcl8.6'),
        os.path.join(bundle_dir, 'lib', 'tcl8.6'),
    ]
    tk_dirs = [
        os.path.join(bundle_dir, 'tk'),
        os.path.join(bundle_dir, 'tk8.6'),
        os.path.join(bundle_dir, 'lib', 'tk8.6'),
    ]
    
    for tcl_path in tcl_dirs:
        if os.path.isdir(tcl_path):
            os.environ['TCL_LIBRARY'] = tcl_path
            break
    
    for tk_path in tk_dirs:
        if os.path.isdir(tk_path):
            os.environ['TK_LIBRARY'] = tk_path
            break
    
    # Add bundle dir to Python path
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

# Run setup immediately when this hook is loaded
setup_environment()
