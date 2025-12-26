#!/usr/bin/env python3
"""
TruthScript - Utility Functions
Helper functions for file handling and format detection.
"""

import os
from typing import List


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return ['.mp3', '.wav', '.m4a', '.aac', '.mp4', '.mov', '.flac', '.ogg', '.wma']


def is_supported_file(filepath: str) -> bool:
    """Check if file is a supported audio/video format."""
    _, ext = os.path.splitext(filepath.lower())
    return ext in get_supported_extensions()


def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to VTT timestamp format: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
