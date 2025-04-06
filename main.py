import os
import sys
import wx
import wx.adv
import json
import time
import shutil
import threading
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import openai
from openai import OpenAI
import re
import io
import subprocess
import wave
import numpy as np
import librosa
import soundfile as sf
from pyannote.core import Segment, Annotation
import concurrent.futures
from threading import Lock
import hashlib
import pickle

# Check if pydub is available for audio conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Check if pyannote is available for speaker diarization
try:
    import torch
    import pyannote.audio
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    from pyannote.audio import Audio
    from pyannote.core import Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

# Ensure required directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = ["Transcripts", "Summaries"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Global variables
app_name = "Audio Processing App"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"
client = None  # OpenAI client instance

# Configuration Manager
class ConfigManager:
    def __init__(self):
        self.config_file = "config.json"
        self.config = self.load_config()
        
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self.default_config()
        return self.default_config()
    
    def default_config(self):
        return {
            "api_key": "",
            "model": DEFAULT_OPENAI_MODEL,
            "temperature": 0.7,
            "language": "english",  # Default language
            "shown_format_info": False,  # Whether we've shown the format info message
            "pyannote_token": "",  # Add token to config
            "templates": {
                "meeting_notes": "# Meeting Summary\n\n## Participants\n{participants}\n\n## Key Points\n{key_points}\n\n## Action Items\n{action_items}",
                "interview": "# Interview Summary\n\n## Interviewee\n{interviewee}\n\n## Main Topics\n{topics}\n\n## Key Insights\n{insights}",
                "lecture": "# Lecture Summary\n\n## Topic\n{topic}\n\n## Main Points\n{main_points}\n\n## Terminology\n{terminology}"
            }
        }
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_api_key(self):
        return self.config.get("api_key", "")
    
    def set_api_key(self, api_key):
        self.config["api_key"] = api_key
        self.save_config()
    
    def get_model(self):
        return self.config.get("model", DEFAULT_OPENAI_MODEL)
    
    def set_model(self, model):
        self.config["model"] = model
        self.save_config()
    
    def get_temperature(self):
        return self.config.get("temperature", 0.7)
    
    def set_temperature(self, temperature):
        self.config["temperature"] = temperature
        self.save_config()
    
    def get_language(self):
        return self.config.get("language", "english")
    
    def set_language(self, language):
        self.config["language"] = language
        self.save_config()
    
    def get_templates(self):
        return self.config.get("templates", {})
    
    def add_template(self, name, template):
        self.config.setdefault("templates", {})[name] = template
        self.save_config()
    
    def remove_template(self, name):
        if name in self.config.get("templates", {}):
            del self.config["templates"][name]
            self.save_config()
    
    # Add methods to get/set pyannote token
    def get_pyannote_token(self):
        return self.config.get("pyannote_token", "")
    
    def set_pyannote_token(self, token):
        self.config["pyannote_token"] = token
        self.save_config()

# Audio Processing Class
class AudioProcessor:
    def __init__(self, client, update_callback=None, config_manager=None):
        self.client = client
        self.update_callback = update_callback
        self.config_manager = config_manager
        self.transcript = ""
        self.speakers = []
        self.word_by_word = []
        self.speaker_segments = []  # Stores time-aligned speaker segments
        self.diarization = None  # Will store pyannote diarization results
        
    def update_status(self, message, percent=None):
        """Update the status with an optional progress percentage."""
        if self.update_callback:
            self.update_callback(message, percent)
            
    def validate_audio_file(self, file_path):
        """Validate that the audio file is suitable for transcription."""
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Define supported formats according to OpenAI API
        supported_formats = ['.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm']
        
        if file_ext not in supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats are: {', '.join(supported_formats)}")
            
        # Check file size (Whisper API limit is 25MB)
        file_size = os.path.getsize(file_path)
        if file_size > 25 * 1024 * 1024:
            raise ValueError(f"File size ({file_size/1024/1024:.2f}MB) exceeds the 25MB limit for the Whisper API")
            
        # Check if file is empty
        if file_size == 0:
            raise ValueError("Audio file is empty")
            
        return True
            
    def convert_to_wav(self, file_path):
        """Convert audio file to WAV format for processing with PyAnnote."""
        output_path = os.path.splitext(file_path)[0] + "_converted.wav"
        
        self.update_status(f"Converting {os.path.basename(file_path)} to WAV format...", percent=0.1)
        
        # Try using pydub if available (handles more formats)
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(file_path)
                audio.export(output_path, format="wav")
                self.update_status("Conversion complete using pydub.", percent=1.0)
                return output_path
            except Exception as e:
                # Fallback to FFmpeg if pydub fails
                self.update_status("Pydub requires FFmpeg to be installed. Trying direct FFmpeg...", percent=0.2)
        
        # Direct FFmpeg conversion
        if not self._is_ffmpeg_available():
            install_instructions = self._get_ffmpeg_install_instructions()
            raise ValueError(f"FFmpeg is required for audio conversion but not found. Please install it.\n\n{install_instructions}")
        
        # Proceed with FFmpeg conversion
        self.update_status("Converting with FFmpeg...", percent=0.5)
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', file_path, 
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                output_path
            ], check=True, stderr=subprocess.PIPE)
            
            self.update_status("Conversion complete using FFmpeg.", percent=1.0)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise ValueError(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
            
    def _get_ffmpeg_install_instructions(self):
        """Return platform-specific instructions for installing FFmpeg."""
        if sys.platform == 'darwin':  # macOS
            return "brew install ffmpeg  (using Homebrew) or visit https://ffmpeg.org/download.html"
        elif sys.platform == 'win32':  # Windows
            return "Download from https://ffmpeg.org/download.html or install using Chocolatey: choco install ffmpeg"
        else:  # Linux
            return "sudo apt install ffmpeg  (Debian/Ubuntu) or sudo yum install ffmpeg (Fedora/CentOS)"
            
    def transcribe_audio(self, file_path, language="en"):
        """Transcribe audio using OpenAI Whisper API."""
        # Validate the audio file
        self.validate_audio_file(file_path)
        
        # Store the file path for potential diarization later
        self.audio_file_path = file_path
        
        # Get file info for status updates
        file_ext = os.path.splitext(file_path)[1].lower()
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Update status with file info
        self.update_status(f"Processing {os.path.basename(file_path)} ({file_ext} format, {file_size_mb:.2f}MB)", percent=0.1)
        
        # Start transcription
        self.update_status(f"Transcribing audio file: {os.path.basename(file_path)}", percent=0.2)
        
        try:
            # Open the file and send to API
            with open(file_path, "rb") as audio_file:
                self.update_status("Sending file to OpenAI Whisper API...", percent=0.3)
                
                try:
                    # First try direct transcription
                    response = self.client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                        language=language,
                        response_format="verbose_json"
                    )
                except Exception as e:
                    error_msg = str(e)
                    # Handle M4A format issues by converting first
                    if file_ext == '.m4a' and ("Invalid file format" in error_msg or "ffprobe" in error_msg):
                        self.update_status("M4A format issue detected. Trying with conversion...", percent=0.4)
                        converted_file = self.convert_to_wav(file_path)
                        
                        with open(converted_file, "rb") as converted_audio:
                            self.update_status("Sending converted file to OpenAI Whisper API...", percent=0.5)
                            response = self.client.audio.transcriptions.create(
                                model=WHISPER_MODEL,
                                file=converted_audio,
                                language=language,
                                response_format="verbose_json"
                            )
                            
                        # Clean up converted file
                        if os.path.exists(converted_file):
                            os.unlink(converted_file)
                    else:
                        # For other errors, re-raise
                        raise
                    
            # Extract transcript text
            transcript_text = response.text
            
            # Store word-by-word data for potential diarization
            if hasattr(response, "words"):
                self.word_by_word = response.words
            
            # Save transcript to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Transcripts/transcript_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            
            # Store transcript for later use
            self.transcript = transcript_text
            
            self.update_status("Transcription complete.", percent=1.0)
            return transcript_text
            
        except openai.AuthenticationError:
            error_msg = "Authentication error. Please check your OpenAI API key."
            self.update_status(error_msg, percent=0)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            self.update_status(error_msg, percent=0)
            raise
            
    def _is_ffmpeg_available(self):
        """Check if ffmpeg is available on the system."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
            
    def identify_speakers(self, transcript):
        """Use optimized methods to identify different speakers in the transcript."""
        self.update_status("Identifying speakers...", percent=0.05)
        
        # Fast path: For very long transcripts without audio, use text-only analysis
        if len(transcript) > 20000 and (not hasattr(self, 'audio_file_path') or not self.audio_file_path):
            self.update_status("Long transcript detected without audio. Using optimized text-only analysis...", percent=0.1)
            return self.identify_speakers_simple(transcript)
        
        # Check if we have the audio file and PyAnnote is available
        if hasattr(self, 'audio_file_path') and self.audio_file_path and PYANNOTE_AVAILABLE:
            try:
                return self.identify_speakers_with_diarization(self.audio_file_path, transcript)
            except Exception as e:
                self.update_status(f"Diarization error: {str(e)}. Falling back to text-based analysis.", percent=0.1)
                return self.identify_speakers_simple(transcript)
        else:
            # Fall back to text-based approach
            return self.identify_speakers_simple(transcript)
            
    def _check_diarization_cache(self, audio_file_path):
        """Check if we have cached diarization results for this file."""
        # Create hash of file path and modification time to use as cache key
        file_stats = os.stat(audio_file_path)
        file_hash = hashlib.md5(f"{audio_file_path}_{file_stats.st_mtime}".encode()).hexdigest()
        
        # Check if cache directory exists
        cache_dir = "diarization_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Cache file path
        cache_file = os.path.join(cache_dir, f"{file_hash}.diar")
        
        # Check if cache file exists
        if os.path.exists(cache_file):
            try:
                self.update_status("Found cached diarization results, loading...", percent=0.15)
                with open(cache_file, 'rb') as f:
                    self.diarization = pickle.load(f)
                self.update_status("Successfully loaded cached diarization results", percent=0.3)
                return True
            except Exception as e:
                self.update_status(f"Error loading cached results: {str(e)}, will reprocess", percent=0.1)
                # If any error occurs, we'll reprocess
                return False
        
        return False
        
    def _save_diarization_cache(self, audio_file_path):
        """Save diarization results to cache."""
        if not hasattr(self, 'diarization') or not self.diarization:
            return
            
        try:
            # Create hash of file path and modification time to use as cache key
            file_stats = os.stat(audio_file_path)
            file_hash = hashlib.md5(f"{audio_file_path}_{file_stats.st_mtime}".encode()).hexdigest()
            
            # Check if cache directory exists
            cache_dir = "diarization_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            # Cache file path
            cache_file = os.path.join(cache_dir, f"{file_hash}.diar")
            
            # Save results
            self.update_status("Saving diarization results to cache for future use...", percent=0.95)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.diarization, f)
            
            # Clean up old cache files if there are more than 20
            cache_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.diar')]
            if len(cache_files) > 20:
                # Sort by modification time and remove oldest
                cache_files.sort(key=os.path.getmtime)
                for old_file in cache_files[:-20]:  # Keep the 20 most recent
                    os.unlink(old_file)
                    
            self.update_status("Successfully cached results for future use", percent=0.98)
        except Exception as e:
            self.update_status(f"Error saving to cache: {str(e)}", percent=0.95)
            # Continue without caching - non-critical error
    
    def identify_speakers_with_diarization(self, audio_file_path, transcript):
        """Identify speakers using audio diarization with PyAnnote."""
        self.update_status("Performing audio diarization analysis...", percent=0.05)
        
        # Check if PyAnnote is available
        if not PYANNOTE_AVAILABLE:
            self.update_status("PyAnnote not available. Install with: pip install pyannote.audio", percent=0)
            return self.identify_speakers_simple(transcript)
        
        # Check if we have cached results - if so, skip to mapping
        if self._check_diarization_cache(audio_file_path):
            self.update_status("Using cached diarization results...", percent=0.4)
            
            # Get audio information for status reporting
            audio_duration = librosa.get_duration(path=audio_file_path)
            is_short_file = audio_duration < 300
            
            # Skip to mapping step
            if is_short_file:
                self.update_status("Fast mapping diarization to transcript...", percent=0.8)
                return self._fast_map_diarization(transcript)
            else:
                return self._map_diarization_to_transcript(transcript)
        
        # No cache, proceed with normal processing
        # Step 1: Initialize PyAnnote pipeline
        try:
            # Get token from config_manager if available
            token = None
            if self.config_manager:
                token = self.config_manager.get_pyannote_token()
            
            # If not found, check for a token file as a fallback
            if not token:
                token_file = "pyannote_token.txt"
                if os.path.exists(token_file):
                    with open(token_file, "r") as f:
                        file_token = f.read().strip()
                        if not file_token.startswith("#") and len(file_token) >= 10:
                            token = file_token
            
            # If still no token, show message and fall back to text-based identification
            if not token:
                self.update_status("PyAnnote token not found in settings. Please add your token in the Settings tab.", percent=0)
                return self.identify_speakers_simple(transcript)
            
            self.update_status("Initializing diarization pipeline...", percent=0.1)
            
            # Initialize the PyAnnote pipeline
            pipeline = pyannote.audio.Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=token
            )
            
            # Set device (GPU if available)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline = pipeline.to(torch.device(device))
            
            # Convert the file to WAV format if needed
            if not audio_file_path.lower().endswith('.wav'):
                self.update_status("Converting audio to WAV format for diarization...", percent=0.15)
                converted_file = self.convert_to_wav(audio_file_path)
                diarization_file = converted_file
            else:
                diarization_file = audio_file_path
            
            # Get audio file information
            audio_duration = librosa.get_duration(path=diarization_file)
            self.update_status(f"Audio duration: {audio_duration:.1f} seconds", percent=0.2)
            
            # Very short files need very different processing approach
            is_short_file = audio_duration < 300  # Less than 5 minutes
            
            if is_short_file:
                # Ultra fast mode for short files (5 min or less) - direct processing with optimized parameters
                self.update_status("Short audio detected, using ultra-fast mode...", percent=0.25)
                
                # Use ultra-optimized parameters for short files
                pipeline.instantiate({
                    # More aggressive voice activity detection for speed
                    "segmentation": {
                        "min_duration_on": 0.25,      # Shorter minimum speech (default 0.1s)
                        "min_duration_off": 0.25,     # Shorter minimum silence (default 0.1s)
                    },
                    # Faster clustering with fewer speakers expected in short clips
                    "clustering": {
                        "min_cluster_size": 6,        # Require fewer samples (default 15)
                        "method": "centroid"          # Faster than "average" linkage
                    },
                    # Skip post-processing for speed
                    "segmentation_batch_size": 32,    # Larger batch for speed
                    "embedding_batch_size": 32,       # Larger batch for speed
                })
                
                # Apply diarization directly for short files
                self.update_status("Processing audio (fast mode)...", percent=0.3)
                self.diarization = pipeline(diarization_file)
                
                # For very short files, optimize the diarization results
                if audio_duration < 60:  # Less than 1 minute
                    # Further optimize by limiting max speakers for very short clips
                    num_speakers = len(set(s for _, _, s in self.diarization.itertracks(yield_label=True)))
                    if num_speakers > 3:
                        self.update_status("Optimizing speaker count for short clip...", percent=0.7)
                        # Re-run with max_speakers=3 for very short clips
                        self.diarization = pipeline(diarization_file, num_speakers=3)
            else:
                # Determine chunk size based on audio duration - longer files use chunking
                if audio_duration > 10800:  # > 3 hours
                    # For extremely long recordings, use very small 3-minute chunks
                    MAX_CHUNK_DURATION = 180  # 3 minutes per chunk
                    self.update_status("Extremely long audio detected (>3 hours). Using highly optimized micro-chunks.", percent=0.22)
                elif audio_duration > 5400:  # > 1.5 hours
                    # For very long recordings, use 4-minute chunks
                    MAX_CHUNK_DURATION = 240  # 4 minutes per chunk
                    self.update_status("Very long audio detected (>1.5 hours). Using micro-chunks for improved performance.", percent=0.22)
                elif audio_duration > 3600:  # > 1 hour
                    # For long recordings, use 5-minute chunks
                    MAX_CHUNK_DURATION = 300  # 5 minutes per chunk
                    self.update_status("Long audio detected (>1 hour). Using optimized chunk size.", percent=0.22)
                elif audio_duration > 1800:  # > 30 minutes
                    # For medium recordings, use 7.5-minute chunks
                    MAX_CHUNK_DURATION = 450  # 7.5 minutes per chunk
                    self.update_status("Medium-length audio detected (>30 minutes). Using optimized chunk size.", percent=0.22)
                else:
                    # Default 10-minute chunks for shorter files
                    MAX_CHUNK_DURATION = 600  # 10 minutes per chunk
                
                # Process in chunks for longer files
                self.update_status("Processing in chunks for optimized performance...", percent=0.25)
                self.diarization = self._process_audio_in_chunks(pipeline, diarization_file, audio_duration, MAX_CHUNK_DURATION)
            
            # Clean up converted file if needed
            if diarization_file != audio_file_path and os.path.exists(diarization_file):
                os.unlink(diarization_file)
            
            # Save diarization results to cache for future use
            self._save_diarization_cache(audio_file_path)
            
            # Now we have diarization data, map it to the transcript using word timestamps
            # Use optimized mapping for short files
            if is_short_file:
                self.update_status("Fast mapping diarization to transcript...", percent=0.8)
                return self._fast_map_diarization(transcript)
            else:
                return self._map_diarization_to_transcript(transcript)
            
        except Exception as e:
            self.update_status(f"Error in diarization: {str(e)}", percent=0)
            # Fall back to text-based approach
            return self.identify_speakers_simple(transcript)

    def _fast_map_diarization(self, transcript):
        """Simplified and faster mapping for short files."""
        self.update_status("Fast mapping diarization results to transcript...", percent=0.85)
        
        if not hasattr(self, 'word_by_word') or not self.word_by_word or not self.diarization:
            return self.identify_speakers_simple(transcript)
        
        try:
            # Create speaker timeline map at higher granularity (every 0.2s)
            timeline_map = {}
            speaker_set = set()
            
            # Extract all speakers and their time ranges
            for segment, _, speaker in self.diarization.itertracks(yield_label=True):
                start_time = segment.start
                end_time = segment.end
                speaker_set.add(speaker)
                
                # For short files, we can afford fine-grained sampling
                step = 0.1  # 100ms steps
                for t in np.arange(start_time, end_time, step):
                    timeline_map[round(t, 1)] = speaker
            
            # Create paragraphs if they don't exist
            if hasattr(self, 'speaker_segments') and self.speaker_segments:
                paragraphs = self.speaker_segments
            else:
                paragraphs = self._create_improved_paragraphs(transcript)
                self.speaker_segments = paragraphs
            
            # Calculate overall speakers - short clips typically have 1-3 speakers
            num_speakers = len(speaker_set)
            self.update_status(f"Detected {num_speakers} speakers in audio", percent=0.9)
            
            # Map each word to a speaker
            word_speakers = {}
            for word_info in self.word_by_word:
                if not hasattr(word_info, "start") or not hasattr(word_info, "end"):
                    continue
                
                # Take the middle point of each word
                word_time = round((word_info.start + word_info.end) / 2, 1)
                
                # Find closest time in our map
                closest_time = min(timeline_map.keys(), key=lambda x: abs(x - word_time), default=None)
                if closest_time is not None and abs(closest_time - word_time) < 1.0:
                    word_speakers[word_info.word] = timeline_map[closest_time]
            
            # Now assign speakers to paragraphs based on word majority
            self.speakers = []
            for paragraph in paragraphs:
                para_speakers = []
                
                # Count speakers in this paragraph
                words = re.findall(r'\b\w+\b', paragraph.lower())
                for word in words:
                    if word in word_speakers:
                        para_speakers.append(word_speakers[word])
                
                # Find most common speaker
                if para_speakers:
                    from collections import Counter
                    speaker_counts = Counter(para_speakers)
                    most_common_speaker = speaker_counts.most_common(1)[0][0]
                    speaker_id = f"Speaker {most_common_speaker.split('_')[-1]}"
                else:
                    # Fallback for paragraphs with no identified speaker
                    speaker_id = f"Speaker 1"
                
                self.speakers.append({
                    "speaker": speaker_id,
                    "text": paragraph
                })
            
            # Final quick consistency check for short files
            if len(self.speakers) > 1:
                self._quick_consistency_check()
            
            self.update_status(f"Diarization complete. Found {num_speakers} speakers.", percent=1.0)
            return self.speakers
            
        except Exception as e:
            self.update_status(f"Error in fast mapping: {str(e)}", percent=0)
            return self.identify_speakers_simple(transcript)
    
    def _map_diarization_to_transcript(self, transcript):
        """Memory-efficient mapping for long files by using sparse sampling and batch processing."""
        self.update_status("Mapping diarization results to transcript (optimized for long files)...", percent=0.8)
        
        if not hasattr(self, 'word_by_word') or not self.word_by_word or not self.diarization:
            return self.identify_speakers_simple(transcript)
            
        try:
            # Get initial speaker count for progress reporting
            speaker_set = set()
            segment_count = 0
            
            # Quick scan to count speakers and segments - don't store details yet
            for segment, _, speaker in self.diarization.itertracks(yield_label=True):
                speaker_set.add(speaker)
                segment_count += 1
                
            num_speakers = len(speaker_set)
            self.update_status(f"Detected {num_speakers} speakers across {segment_count} segments", percent=0.82)
            
            # Create paragraphs if they don't exist
            if hasattr(self, 'speaker_segments') and self.speaker_segments:
                paragraphs = self.speaker_segments
            else:
                paragraphs = self._create_improved_paragraphs(transcript)
                self.speaker_segments = paragraphs
                
            # OPTIMIZATION 1: For long files, use sparse sampling of the timeline
            # Instead of creating a dense timeline map which is memory-intensive,
            # we'll create a sparse map with only the segment boundaries
            timeline_segments = []
            
            # Use diarization_cache directory for temporary storage if needed
            cache_dir = "diarization_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            # OPTIMIZATION 2: For very long files, process diarization in chunks to avoid memory issues
            chunk_size = 1000  # Process 1000 segments at a time
            use_temp_storage = segment_count > 5000  # Only use temp storage for very large files
            
            # If using temp storage, save intermediate results to avoid memory buildup
            if use_temp_storage:
                self.update_status("Using temporary storage for large diarization data...", percent=0.83)
                temp_file = os.path.join(cache_dir, f"diarization_map_{int(time.time())}.json")
                
                # Process in chunks to avoid memory buildup
                processed = 0
                segment_chunk = []
                
                for segment, _, speaker in self.diarization.itertracks(yield_label=True):
                    # Skip very short segments
                    if segment.duration < 0.5:
                        continue
                        
                    segment_chunk.append({
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": speaker
                    })
                    
                    processed += 1
                    
                    # When chunk is full, process it
                    if len(segment_chunk) >= chunk_size:
                        timeline_segments.extend(segment_chunk)
                        # Save intermediate results
                        with open(temp_file, 'w') as f:
                            json.dump(timeline_segments, f)
                        # Clear memory
                        timeline_segments = []
                        segment_chunk = []
                        # Update progress
                        progress = 0.83 + (processed / segment_count) * 0.05
                        self.update_status(f"Processed {processed}/{segment_count} diarization segments...", percent=progress)
                
                # Process remaining segments
                if segment_chunk:
                    timeline_segments.extend(segment_chunk)
                    with open(temp_file, 'w') as f:
                        json.dump(timeline_segments, f)
                
                # Load from file to continue processing
                with open(temp_file, 'r') as f:
                    timeline_segments = json.load(f)
            else:
                # For smaller files, process all at once
                for segment, _, speaker in self.diarization.itertracks(yield_label=True):
                    # Skip very short segments
                    if segment.duration < 0.5:
                        continue
                        
                    timeline_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": speaker
                    })
            
            self.update_status("Matching words to speaker segments...", percent=0.89)
            
            # OPTIMIZATION 3: Optimize word-to-speaker mapping for long files
            # Sort segments by start time for faster searching
            timeline_segments.sort(key=lambda x: x["start"])
            
            # Initialize paragraph mapping structures
            paragraph_speaker_counts = [{} for _ in paragraphs]
            
            # Batch process words to reduce computation
            batch_size = 500
            num_words = len(self.word_by_word)
            
            # Calculate which paragraph each word belongs to
            word_paragraphs = {}
            
            para_start_idx = 0
            for i, word_info in enumerate(self.word_by_word):
                if not hasattr(word_info, "start") or not hasattr(word_info, "end"):
                    continue
                    
                # Binary search to find the paragraph for this word
                # This is much faster than iterating through all paragraphs for each word
                word = word_info.word.lower()
                
                # Find paragraph for this word only once
                if i % 100 == 0:  # Only update progress occasionally
                    progress = 0.89 + (i / num_words) * 0.05
                    self.update_status(f"Matching words to paragraphs ({i}/{num_words})...", percent=progress)
                
                # Find which paragraph this word belongs to
                found_para = False
                for p_idx in range(para_start_idx, len(paragraphs)):
                    if word in paragraphs[p_idx].lower():
                        word_paragraphs[word] = p_idx
                        para_start_idx = p_idx  # Optimization: start next search from here
                        found_para = True
                        break
                
                if not found_para:
                    # If we didn't find it moving forward, try searching all paragraphs
                    for p_idx in range(len(paragraphs)):
                        if word in paragraphs[p_idx].lower():
                            word_paragraphs[word] = p_idx
                            para_start_idx = p_idx
                            found_para = True
                            break
            
            # Process words in batches to assign speakers efficiently
            for batch_start in range(0, num_words, batch_size):
                batch_end = min(batch_start + batch_size, num_words)
                
                for i in range(batch_start, batch_end):
                    if i >= len(self.word_by_word):
                        break
                        
                    word_info = self.word_by_word[i]
                    if not hasattr(word_info, "start") or not hasattr(word_info, "end"):
                        continue
                    
                    word = word_info.word.lower()
                    word_time = (word_info.start + word_info.end) / 2
                    
                    # Find segment for this word using binary search for speed
                    left, right = 0, len(timeline_segments) - 1
                    segment_idx = -1
                    
                    while left <= right:
                        mid = (left + right) // 2
                        if timeline_segments[mid]["start"] <= word_time <= timeline_segments[mid]["end"]:
                            segment_idx = mid
                            break
                        elif word_time < timeline_segments[mid]["start"]:
                            right = mid - 1
                        else:
                            left = mid + 1
                    
                    # If we found a segment, update the paragraph speaker counts
                    if segment_idx != -1:
                        speaker = timeline_segments[segment_idx]["speaker"]
                        
                        # If we know which paragraph this word belongs to, update its speaker count
                        if word in word_paragraphs:
                            para_idx = word_paragraphs[word]
                            paragraph_speaker_counts[para_idx][speaker] = paragraph_speaker_counts[para_idx].get(speaker, 0) + 1
                
                # Update progress
                progress = 0.94 + (batch_end / num_words) * 0.05
                self.update_status(f"Processed {batch_end}/{num_words} words...", percent=progress)
            
            # Assign speakers to paragraphs based on majority vote
            self.speakers = []
            for i, paragraph in enumerate(paragraphs):
                # Get speaker counts for this paragraph
                speaker_counts = paragraph_speaker_counts[i]
                
                # Assign the most common speaker, or default if none
                if speaker_counts:
                    # Find speaker with highest count
                    most_common_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
                    speaker_id = f"Speaker {most_common_speaker.split('_')[-1]}"
                else:
                    # Default speaker if no match found
                    speaker_id = f"Speaker 1"
                
                self.speakers.append({
                    "speaker": speaker_id,
                    "text": paragraph
                })
            
            # Quick consistency check
            if len(self.speakers) > 2:
                self._quick_consistency_check()
            
            # Clean up temp file if used
            if use_temp_storage and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            self.update_status(f"Diarization mapping complete. Found {num_speakers} speakers.", percent=1.0)
            return self.speakers
            
        except Exception as e:
            self.update_status(f"Error in diarization mapping: {str(e)}", percent=0)
            # Fall back to text-based approach
            return self.identify_speakers_simple(transcript)
    
    def _quick_consistency_check(self):
        """Ultra-quick consistency check for short files"""
        if len(self.speakers) < 3:
            return
            
        # Look for isolated speaker segments
        for i in range(1, len(self.speakers) - 1):
            prev_speaker = self.speakers[i-1]["speaker"]
            curr_speaker = self.speakers[i]["speaker"]
            next_speaker = self.speakers[i+1]["speaker"]
            
            # If current speaker is sandwiched between different speakers
            if prev_speaker == next_speaker and curr_speaker != prev_speaker:
                # Fix the segment only if very short (likely error)
                if len(self.speakers[i]["text"].split()) < 15:
                    self.speakers[i]["speaker"] = prev_speaker

    def _process_audio_in_chunks(self, pipeline, audio_file, total_duration, chunk_size):
        """Process long audio files in chunks to optimize memory usage and speed."""
        from pyannote.core import Segment, Annotation
        import concurrent.futures
        from threading import Lock
        
        # Initialize a combined annotation object
        combined_diarization = Annotation()
        
        # Calculate number of chunks
        num_chunks = int(np.ceil(total_duration / chunk_size))
        self.update_status(f"Processing audio in {num_chunks} chunks...", percent=0.1)
        
        # Optimize number of workers based on file length and available memory
        # More chunks = more workers (up to cpu_count), but limit for very long files
        # to avoid excessive memory usage
        cpu_count = os.cpu_count() or 4
        
        # Try to get available system memory if psutil is available
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            # Scale workers based on available memory - each worker needs ~500MB-1GB
            memory_based_workers = max(1, int(available_memory_gb / 1.5))
        except ImportError:
            # If psutil is not available, make a conservative estimate
            memory_based_workers = 4
            
        # Scale workers based on file duration
        if audio_duration > 10800:  # > 3 hours
            # For extremely long files, be very conservative with worker count
            duration_based_workers = min(3, cpu_count)
        elif audio_duration > 5400:  # > 1.5 hours
            # For very long files, be conservative
            duration_based_workers = min(4, cpu_count)
        elif audio_duration > 3600:  # > 1 hour
            # For long files
            duration_based_workers = min(6, cpu_count)
        else:
            # For shorter files we can use more workers
            duration_based_workers = min(8, cpu_count)
            
        # Take the minimum of the two estimates
        max_workers = min(memory_based_workers, duration_based_workers)
        # Ensure at least one worker
        max_workers = max(1, max_workers)
        
        self.update_status(f"Using {max_workers} parallel workers for processing...", percent=0.12)
        
        # Lock for thread-safe updates
        lock = Lock()
        result_counter = [0]  # Use a list so we can modify it from the worker
        
        # Define a worker function to process a chunk with optimized parameters
        def process_chunk(chunk_info):
            i, start_time, end_time = chunk_info
            chunk_result = Annotation()
            
            # Create temporary file for this chunk
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                chunk_path = tmp_file.name
            
            try:
                # Extract chunk using ffmpeg with optimized parameters
                # -threads 2: Use 2 threads per extraction process
                # -ac 1: Convert to mono
                # -ar 16000: Use 16kHz sample rate (sufficient for voice)
                subprocess.run([
                    'ffmpeg', '-y', '-loglevel', 'error', '-threads', '2',
                    '-i', audio_file,
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    chunk_path
                ], check=True)
                
                # Process this chunk
                chunk_diarization = pipeline(chunk_path)
                
                # Adjust timestamps and add to result - only include speakers that talk for >0.5s
                min_speech_duration = 0.5  # Filter out very short segments
                
                for segment, track, speaker in chunk_diarization.itertracks(yield_label=True):
                    # Skip very short segments (often noise or artifacts)
                    if segment.duration < min_speech_duration:
                        continue
                        
                    adjusted_start = segment.start + start_time
                    adjusted_end = segment.end + start_time
                    adjusted_segment = Segment(adjusted_start, adjusted_end)
                    chunk_result[adjusted_segment, track] = speaker
                
                # Update progress inside lock
                with lock:
                    result_counter[0] += 1
                    progress = (result_counter[0] / num_chunks) * 0.8 + 0.1
                    self.update_status(f"Processed chunk {result_counter[0]}/{num_chunks} ({start_time:.1f}s to {end_time:.1f}s)...", 
                                      percent=progress)
                
                return chunk_result
                
            except Exception as e:
                self.update_status(f"Error processing chunk {i+1}: {str(e)}", percent=0.1)
                return Annotation()  # Return empty annotation on error
            finally:
                # Clean up temporary file
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
        
        # Create chunk information - include 1-second overlap between chunks
        chunk_infos = []
        for i in range(num_chunks):
            start_time = max(0, i * chunk_size - 1) if i > 0 else 0
            end_time = min((i + 1) * chunk_size + 1, total_duration) if i < num_chunks - 1 else total_duration
            chunk_infos.append((i, start_time, end_time))
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(process_chunk, chunk_info): chunk_info 
                for chunk_info in chunk_infos
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_result = future.result()
                
                # Merge chunk result into combined result
                with lock:
                    for segment, track, speaker in chunk_result.itertracks(yield_label=True):
                        combined_diarization[segment, track] = speaker
        
        self.update_status(f"Completed processing all {num_chunks} chunks.", percent=0.9)
        
        # Clean up the annotation - join nearby segments from the same speaker
        self.update_status("Post-processing and optimizing results...", percent=0.92)
        combined_diarization = self._optimize_diarization_results(combined_diarization)
        
        return combined_diarization
        
    def _optimize_diarization_results(self, diarization):
        """Optimize diarization results by joining nearby segments from the same speaker."""
        from pyannote.core import Segment, Annotation
        
        # Create a new annotation for the optimized result
        optimized = Annotation()
        
        # Group by speaker
        for speaker in diarization.labels():
            speaker_turns = list(diarization.label_timeline(speaker))
            
            # Join segments that are close to each other (less than 0.5s gap)
            max_gap = 0.5
            i = 0
            while i < len(speaker_turns):
                current = speaker_turns[i]
                j = i + 1
                while j < len(speaker_turns) and speaker_turns[j].start - speaker_turns[j-1].end <= max_gap:
                    current = Segment(current.start, speaker_turns[j].end)
                    j += 1
                
                # Add the joined segment to the optimized annotation
                optimized[current] = speaker
                i = j
        
        return optimized
    
    def _identify_speakers_chunked(self, paragraphs, chunk_size):
        """Process long transcripts in chunks for speaker identification."""
        self.update_status("Processing transcript in chunks...", percent=0.1)
        
        # Group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for p in paragraphs:
            if current_length + len(p) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [p]
                current_length = len(p)
            else:
                current_chunk.append(p)
                current_length += len(p)
                
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        self.update_status(f"Processing transcript in {len(chunks)} chunks...", percent=0.15)
        
        # Process first chunk to establish speaker patterns
        model_to_use = DEFAULT_OPENAI_MODEL
        
        # Initialize result container
        all_results = []
        speaker_characteristics = {}
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Calculate progress percentage (0-1)
            progress = (i / len(chunks)) * 0.7 + 0.2  # 20% to 90% of total progress
            
            self.update_status(f"Processing chunk {i+1}/{len(chunks)}...", percent=progress)
            
            # For first chunk, get detailed analysis
            if i == 0:
                prompt = f"""
                Analyze this transcript segment and identify exactly two speakers (A and B).
                
                TASK:
                1. Determine which paragraphs belong to which speaker
                2. Identify each speaker's characteristics and speaking style
                3. Ensure logical conversation flow
                
                Return JSON in this exact format:
                {{
                    "analysis": {{
                        "speaker_a_characteristics": ["characteristic 1", "characteristic 2"],
                        "speaker_b_characteristics": ["characteristic 1", "characteristic 2"]
                    }},
                    "paragraphs": [
                        {{
                            "id": {len(all_results)},
                            "speaker": "A",
                            "text": "paragraph text"
                        }},
                        ...
                    ]
                }}
                
                Transcript paragraphs:
                {json.dumps([{"id": len(all_results) + j, "text": p} for j, p in enumerate(chunk)])}
                """
            else:
                # For subsequent chunks, use characteristics from first analysis
                prompt = f"""
                Continue assigning speakers to this transcript segment.
                
                Speaker A characteristics: {json.dumps(speaker_characteristics.get("speaker_a_characteristics", []))}
                Speaker B characteristics: {json.dumps(speaker_characteristics.get("speaker_b_characteristics", []))}
                
                Return JSON with speaker assignments:
                {{
                    "paragraphs": [
                        {{
                            "id": {len(all_results)},
                            "speaker": "A or B",
                            "text": "paragraph text"
                        }},
                        ...
                    ]
                }}
                
                Transcript paragraphs:
                {json.dumps([{"id": len(all_results) + j, "text": p} for j, p in enumerate(chunk)])}
                """
            
            # Make API call for this chunk
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are an expert conversation analyst who identifies speaker turns in transcripts with high accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Save speaker characteristics from first chunk
            if i == 0 and "analysis" in result:
                speaker_characteristics = result["analysis"]
            
            # Add results from this chunk
            if "paragraphs" in result:
                all_results.extend(result["paragraphs"])
            
            # Update progress
            after_progress = (i + 0.5) / len(chunks) * 0.7 + 0.2
            self.update_status(f"Processed chunk {i+1}/{len(chunks)}...", percent=after_progress)
        
        # Map Speaker A/B to Speaker 1/2
        speaker_map = {
            "A": "Speaker 1", 
            "B": "Speaker 2",
            "Speaker A": "Speaker 1", 
            "Speaker B": "Speaker 2"
        }
        
        self.update_status("Finalizing speaker assignments...", percent=0.95)
        
        # Create final speakers list
        self.speakers = []
        for item in sorted(all_results, key=lambda x: x.get("id", 0)):
            speaker_label = item.get("speaker", "Unknown")
            mapped_speaker = speaker_map.get(speaker_label, speaker_label)
            
            self.speakers.append({
                "speaker": mapped_speaker,
                "text": item.get("text", "")
            })
        
        # Ensure we have the right number of paragraphs
        if len(self.speakers) != len(paragraphs):
            self.update_status(f"Warning: Received {len(self.speakers)} segments but expected {len(paragraphs)}. Fixing...", percent=0.98)
            self.speakers = [
                {"speaker": self.speakers[min(i, len(self.speakers)-1)]["speaker"] if self.speakers else f"Speaker {i % 2 + 1}", 
                 "text": p}
                for i, p in enumerate(paragraphs)
            ]
        
        self.update_status(f"Speaker identification complete. Found 2 speakers across {len(chunks)} chunks.", percent=1.0)
        return self.speakers

    def identify_speakers_simple(self, transcript):
        """Identify speakers using a simplified and optimized approach."""
        self.update_status("Analyzing transcript for speaker identification...", percent=0.1)
        
        # First, split transcript into paragraphs
        paragraphs = self._create_improved_paragraphs(transcript)
        self.speaker_segments = paragraphs
        
        # Setup model
        model_to_use = DEFAULT_OPENAI_MODEL
        
        # For very long transcripts, we'll analyze in chunks
        MAX_CHUNK_SIZE = 8000  # characters per chunk
        
        if len(transcript) > MAX_CHUNK_SIZE:
            self.update_status("Long transcript detected. Processing in chunks...", percent=0.15)
            return self._identify_speakers_chunked(paragraphs, MAX_CHUNK_SIZE)
        
        # Enhanced single-pass approach for shorter transcripts
        prompt = f"""
        Analyze this transcript and identify exactly two speakers (A and B).
        
        TASK:
        1. Determine which paragraphs belong to which speaker
        2. Focus on conversation pattern and speaking style
        3. Ensure logical conversation flow (e.g., questions are followed by answers)
        4. Maintain consistency in first-person statements
        
        Return JSON in this exact format:
        {{
            "analysis": {{
                "speaker_a_characteristics": ["characteristic 1", "characteristic 2"],
                "speaker_b_characteristics": ["characteristic 1", "characteristic 2"],
                "speaker_count": 2,
                "conversation_type": "interview/discussion/etc"
            }},
            "paragraphs": [
                {{
                    "id": 0,
                    "speaker": "A",
                    "text": "paragraph text"
                }},
                ...
            ]
        }}
        
        Transcript paragraphs:
        {json.dumps([{"id": i, "text": p} for i, p in enumerate(paragraphs)])}
        """
        
        try:
            # Single API call to assign speakers
            self.update_status("Sending transcript for speaker analysis...", percent=0.3)
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are an expert conversation analyst who identifies speaker turns in transcripts with high accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            self.update_status("Processing speaker identification results...", percent=0.7)
            result = json.loads(response.choices[0].message.content)
            
            # Get paragraph assignments
            assignments = result.get("paragraphs", [])
            
            # Map Speaker A/B to Speaker 1/2 for compatibility with existing system
            speaker_map = {
                "A": "Speaker 1", 
                "B": "Speaker 2",
                "Speaker A": "Speaker 1", 
                "Speaker B": "Speaker 2"
            }
            
            # Create speakers list with proper mapping
            self.speakers = []
            for item in sorted(assignments, key=lambda x: x.get("id", 0)):
                speaker_label = item.get("speaker", "Unknown")
                mapped_speaker = speaker_map.get(speaker_label, speaker_label)
                
                self.speakers.append({
                    "speaker": mapped_speaker,
                    "text": item.get("text", "")
                })
            
            # Ensure we have the right number of paragraphs
            if len(self.speakers) != len(paragraphs):
                self.update_status(f"Warning: Received {len(self.speakers)} segments but expected {len(paragraphs)}. Fixing...", percent=0.9)
                self.speakers = [
                    {"speaker": self.speakers[min(i, len(self.speakers)-1)]["speaker"] if self.speakers else f"Speaker {i % 2 + 1}", 
                     "text": p}
                    for i, p in enumerate(paragraphs)
                ]
            
            self.update_status(f"Speaker identification complete. Found {2} speakers.", percent=1.0)
            return self.speakers
            
        except Exception as e:
            self.update_status(f"Error in speaker identification: {str(e)}", percent=0)
            # Fallback to basic alternating speaker assignment
            self.speakers = [
                {"speaker": f"Speaker {i % 2 + 1}", "text": p}
                for i, p in enumerate(paragraphs)
            ]
            return self.speakers
            
    def _create_improved_paragraphs(self, transcript):
        """Create more intelligent paragraph breaks based on semantic analysis."""
        import re
        # Split transcript into sentences
        sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into paragraphs
        paragraphs = []
        current_para = []
        
        # These phrases often signal the start of a new speaker's turn
        new_speaker_indicators = [
            "yes", "no", "I think", "I believe", "so,", "well,", "actually", 
            "to be honest", "in my opinion", "I agree", "I disagree",
            "let me", "I'd like to", "I would", "you know", "um", "uh", 
            "hmm", "but", "however", "from my perspective", "wait", "okay",
            "right", "sure", "exactly", "absolutely", "definitely", "perhaps",
            "look", "listen", "basically", "frankly", "honestly", "now", "so",
            "thank you", "thanks", "good point", "interesting", "true", "correct",
            "first of all", "firstly", "secondly", "finally", "in conclusion"
        ]
        
        # Words/phrases that indicate continuation by the same speaker
        continuation_indicators = [
            "and", "also", "additionally", "moreover", "furthermore", "plus",
            "then", "after that", "next", "finally", "lastly", "in addition",
            "consequently", "as a result", "therefore", "thus", "besides",
            "for example", "specifically", "in particular", "especially",
            "because", "since", "due to", "as such", "which means"
        ]
        
        for i, sentence in enumerate(sentences):
            # Start a new paragraph if:
            start_new_para = False
            
            # 1. This is the first sentence
            if i == 0:
                start_new_para = True
                
            # 2. Previous sentence ended with a question mark
            elif i > 0 and sentences[i-1].endswith('?'):
                start_new_para = True
                
            # 3. Current sentence begins with a common new speaker phrase
            elif any(sentence.lower().startswith(indicator.lower()) for indicator in new_speaker_indicators):
                start_new_para = True
                
            # 4. Not a continuation and not a pronoun reference
            elif (i > 0 and 
                  not any(sentence.lower().startswith(indicator.lower()) for indicator in continuation_indicators) and
                  not re.match(r'^(It|This|That|These|Those|They|He|She|We|I)\b', sentence, re.IGNORECASE) and
                  len(current_para) >= 2):
                start_new_para = True
                
            # 5. Natural length limit to avoid overly long paragraphs
            elif len(current_para) >= 4:
                start_new_para = True
            
            # Start a new paragraph if needed
            if start_new_para and current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
            
            current_para.append(sentence)
        
        # Add the last paragraph
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        return paragraphs

    def assign_speaker_names(self, speaker_map):
        """Apply custom speaker names to the transcript."""
        if not hasattr(self, 'speakers') or not self.speakers:
            return self.transcript
            
        # Create a formatted transcript with the new speaker names
        formatted_text = []
        
        for segment in self.speakers:
            original_speaker = segment.get("speaker", "Unknown")
            new_speaker = speaker_map.get(original_speaker, original_speaker)
            text = segment.get("text", "")
            
            formatted_text.append(f"{new_speaker}: {text}")
            
        return "\n\n".join(formatted_text)

# LLM Processing Class
class LLMProcessor:
    def __init__(self, client, config_manager, update_callback=None):
        self.client = client
        self.config_manager = config_manager
        self.update_callback = update_callback
        self.chat_history = []
        
    def update_status(self, message, percent=None):
        if self.update_callback:
            wx.CallAfter(self.update_callback, message, percent)
            
    def generate_response(self, prompt, temperature=None):
        """Generate a response from the LLM."""
        if temperature is None:
            temperature = self.config_manager.get_temperature()
            
        model = self.config_manager.get_model()
        messages = self.prepare_messages(prompt)
        
        try:
            self.update_status("Generating response...", percent=0)
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            
            response_text = response.choices[0].message.content
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": response_text})
            
            self.update_status("Response generated.", percent=100)
            return response_text
            
        except Exception as e:
            self.update_status(f"Error generating response: {str(e)}", percent=50)
            return f"Error: {str(e)}"
            
    def prepare_messages(self, prompt):
        """Prepare messages for the LLM, including chat history."""
        messages = []
        
        # Add system message
        system_content = "You are a helpful assistant that can analyze transcripts."
        messages.append({"role": "system", "content": system_content})
        
        # Add chat history (limit to last 10 messages to avoid token limits)
        if self.chat_history:
            messages.extend(self.chat_history[-10:])
            
        # Add the current prompt
        if prompt not in [msg["content"] for msg in messages if msg["role"] == "user"]:
            messages.append({"role": "user", "content": prompt})
            
        return messages
        
    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history = []
        self.update_status("Chat history cleared.", percent=0)
        
    def summarize_transcript(self, transcript, template_name=None):
        """Summarize a transcript, optionally using a template."""
        if not transcript:
            return "No transcript to summarize."
            
        self.update_status("Generating summary...", percent=0)
        
        prompt = f"Summarize the following transcript:"
        template = None
        
        if template_name:
            templates = self.config_manager.get_templates()
            if template_name in templates:
                template = templates[template_name]
                prompt += f" Follow this template format:\n\n{template}"
                
        prompt += f"\n\nTranscript:\n{transcript}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.config_manager.get_model(),
                messages=[
                    {"role": "system", "content": "You are an assistant that specializes in summarizing transcripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            summary = response.choices[0].message.content
            
            # Save summary to file
            summary_filename = f"Summaries/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary)
                
            self.update_status(f"Summary generated and saved to {summary_filename}.", percent=100)
            return summary
            
        except Exception as e:
            self.update_status(f"Error generating summary: {str(e)}", percent=50)
            return f"Error: {str(e)}"

# GUI - Main Application Frame
class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MainFrame, self).__init__(parent, title=title, size=(1200, 800))
        
        # Initialize config manager
        self.config_manager = ConfigManager()
        
        # Check for API key and initialize client
        self.initialize_openai_client()
        
        # Initialize processors
        self.audio_processor = AudioProcessor(client, self.update_status, self.config_manager)
        self.llm_processor = LLMProcessor(client, self.config_manager, self.update_status)
        
        # Set up the UI
        self.create_ui()
        
        # Event bindings
        self.bind_events()
        
        # Center the window
        self.Centre()
        
        # Create required directories
        ensure_directories()
        
        # Status update
        self.update_status("Application ready.", percent=0)
        
        # Display info about supported audio formats
        wx.CallLater(1000, self.show_format_info)
        
        # Check for PyAnnote and display installation message if needed
        wx.CallLater(1500, self.check_pyannote)
    
    def initialize_openai_client(self):
        """Initialize OpenAI client with API key."""
        global client
        api_key = self.config_manager.get_api_key()
        
        if not api_key:
            dlg = wx.TextEntryDialog(self, "Please enter your OpenAI API key:", "API Key Required")
            if dlg.ShowModal() == wx.ID_OK:
                api_key = dlg.GetValue()
                self.config_manager.set_api_key(api_key)
            dlg.Destroy()
        
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            wx.MessageBox(f"Error initializing OpenAI client: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
            
    def create_ui(self):
        """Create the user interface."""
        # Create notebook for tabbed interface
        self.notebook = wx.Notebook(self)
        
        # Create panels for each tab
        self.audio_panel = wx.Panel(self.notebook)
        self.chat_panel = wx.Panel(self.notebook)
        self.settings_panel = wx.Panel(self.notebook)
        
        # Add panels to notebook
        self.notebook.AddPage(self.audio_panel, "Audio Processing")
        self.notebook.AddPage(self.chat_panel, "Chat")
        self.notebook.AddPage(self.settings_panel, "Settings")
        
        # Bind the notebook page change event
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_notebook_page_changed)
        
        # Create UI for each panel
        self.create_audio_panel()
        self.create_chat_panel()
        self.create_settings_panel()
        
        # Add status bar
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Ready")
        
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)
        
    def on_notebook_page_changed(self, event):
        """Handle notebook page change event."""
        old_page = event.GetOldSelection()
        new_page = event.GetSelection()
        
        # If user switched from settings to audio tab, update the speaker ID button styling
        if old_page == 2 and new_page == 0:  # 2 = settings, 0 = audio
            self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
            self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
            self.update_speaker_id_button_style()
            self.audio_panel.Layout()
            
        event.Skip()  # Allow default event processing
        
    def create_audio_panel(self):
        """Create the audio processing panel."""
        panel = self.audio_panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # File upload section
        file_box = wx.StaticBox(panel, label="Audio File")
        file_sizer = wx.StaticBoxSizer(file_box, wx.VERTICAL)
        
        file_select_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.audio_file_path = wx.TextCtrl(panel, style=wx.TE_READONLY)
        browse_btn = wx.Button(panel, label="Browse")
        browse_btn.Bind(wx.EVT_BUTTON, self.on_browse_audio)
        
        file_select_sizer.Add(self.audio_file_path, 1, wx.EXPAND | wx.RIGHT, 5)
        file_select_sizer.Add(browse_btn, 0)
        
        file_sizer.Add(file_select_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Language selection
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lang_sizer.Add(wx.StaticText(panel, label="Language:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.language_choice = wx.Choice(panel, choices=["English", "Hungarian"])
        self.language_choice.SetSelection(0 if self.config_manager.get_language() == "english" else 1)
        lang_sizer.Add(self.language_choice, 0, wx.LEFT, 5)
        
        # Transcribe button
        self.transcribe_btn = wx.Button(panel, label="Transcribe")
        self.transcribe_btn.Bind(wx.EVT_BUTTON, self.on_transcribe)
        
        file_sizer.Add(lang_sizer, 0, wx.EXPAND | wx.ALL, 5)
        file_sizer.Add(self.transcribe_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(file_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Transcript display
        transcript_box = wx.StaticBox(panel, label="Transcript")
        transcript_sizer = wx.StaticBoxSizer(transcript_box, wx.VERTICAL)
        
        self.transcript_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        transcript_sizer.Add(self.transcript_text, 1, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(transcript_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        # Speaker identification section
        speaker_box = wx.StaticBox(panel, label="Speaker Identification")
        speaker_sizer = wx.StaticBoxSizer(speaker_box, wx.VERTICAL)
        
        # Bold font for button text
        button_font = wx.Font(wx.NORMAL_FONT.GetPointSize(), wx.FONTFAMILY_DEFAULT, 
                           wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        
        # Create button with appropriate styling based on PyAnnote status
        self.identify_speakers_btn = wx.Button(panel, label=self.get_speaker_id_button_label())
        self.identify_speakers_btn.SetFont(button_font)
        
        # Set button color based on PyAnnote status
        self.update_speaker_id_button_style()
        
        self.identify_speakers_btn.Bind(wx.EVT_BUTTON, self.on_identify_speakers)
        speaker_sizer.Add(self.identify_speakers_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        # Add a help text that will also be updated based on PyAnnote status
        self.speaker_id_help_text = wx.StaticText(panel, label=self.get_speaker_id_help_text())
        speaker_sizer.Add(self.speaker_id_help_text, 0, wx.CENTER | wx.ALL, 5)
        
        # Speaker mapping UI
        self.speaker_mapping_panel = wx.Panel(panel)
        self.speaker_mapping_sizer = wx.FlexGridSizer(cols=2, vgap=5, hgap=5)
        self.speaker_mapping_sizer.AddGrowableCol(1)
        self.speaker_mapping_panel.SetSizer(self.speaker_mapping_sizer)
        
        speaker_sizer.Add(self.speaker_mapping_panel, 0, wx.EXPAND | wx.ALL, 5)
        
        # Apply speaker names button
        self.apply_speaker_names_btn = wx.Button(panel, label="Apply Speaker Names")
        self.apply_speaker_names_btn.Bind(wx.EVT_BUTTON, self.on_apply_speaker_names)
        speaker_sizer.Add(self.apply_speaker_names_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(speaker_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Summarization section
        summary_box = wx.StaticBox(panel, label="Summarization")
        summary_sizer = wx.StaticBoxSizer(summary_box, wx.VERTICAL)
        
        template_sizer = wx.BoxSizer(wx.HORIZONTAL)
        template_sizer.Add(wx.StaticText(panel, label="Template:"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        templates = list(self.config_manager.get_templates().keys())
        self.template_choice = wx.Choice(panel, choices=["None"] + templates)
        self.template_choice.SetSelection(0)
        template_sizer.Add(self.template_choice, 1, wx.LEFT, 5)
        
        summary_sizer.Add(template_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        self.summarize_btn = wx.Button(panel, label="Generate Summary")
        self.summarize_btn.Bind(wx.EVT_BUTTON, self.on_summarize)
        summary_sizer.Add(self.summarize_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(summary_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Set the panel's sizer
        panel.SetSizer(sizer)
        
        # Initial button states
        self.update_button_states()
        
    def get_speaker_id_button_label(self):
        """Get the appropriate label for the speaker identification button."""
        if PYANNOTE_AVAILABLE and self.config_manager.get_pyannote_token():
            return "Identify Speakers using Audio Analysis"
        elif PYANNOTE_AVAILABLE:
            return "Identify Speakers (Token Required)"
        else:
            return "Identify Speakers using Text Analysis"
            
    def get_speaker_id_help_text(self):
        """Get the appropriate help text for speaker identification."""
        if PYANNOTE_AVAILABLE and self.config_manager.get_pyannote_token():
            return "Click above to detect speakers using voice analysis (more accurate)"
        elif PYANNOTE_AVAILABLE:
            return "PyAnnote is installed but requires a token. See Settings tab."
        else:
            return "Click above to detect speakers using text patterns"
            
    def update_speaker_id_button_style(self):
        """Update the style of the speaker identification button based on PyAnnote status."""
        if not hasattr(self, 'identify_speakers_btn'):
            return
            
        if PYANNOTE_AVAILABLE and self.config_manager.get_pyannote_token():
            # PyAnnote available and token configured - green button
            self.identify_speakers_btn.SetBackgroundColour(wx.Colour(200, 255, 200))  # Light green
            self.identify_speakers_btn.SetForegroundColour(wx.Colour(0, 100, 0))      # Dark green
        elif PYANNOTE_AVAILABLE:
            # PyAnnote available but token missing - orange button
            self.identify_speakers_btn.SetBackgroundColour(wx.Colour(255, 224, 178))  # Light orange
            self.identify_speakers_btn.SetForegroundColour(wx.Colour(153, 76, 0))     # Dark orange
        else:
            # PyAnnote not available - light blue button (original style)
            self.identify_speakers_btn.SetBackgroundColour(wx.Colour(220, 230, 255))  # Light blue
            self.identify_speakers_btn.SetForegroundColour(wx.BLACK)
            
    def on_save_pyannote_token(self, event):
        """Save the PyAnnote token."""
        token = self.pyannote_token_input.GetValue()
        self.config_manager.set_pyannote_token(token)
        
        # Update the speaker identification button style
        self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
        self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
        self.update_speaker_id_button_style()
        self.audio_panel.Layout()
        
        wx.MessageBox("PyAnnote token saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def create_chat_panel(self):
        """Create the chat panel."""
        panel = self.chat_panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Chat history display
        chat_box = wx.StaticBox(panel, label="Chat History")
        chat_sizer = wx.StaticBoxSizer(chat_box, wx.VERTICAL)
        
        self.chat_history_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        chat_sizer.Add(self.chat_history_text, 1, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(chat_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        # User input section
        input_box = wx.StaticBox(panel, label="User Input")
        input_sizer = wx.StaticBoxSizer(input_box, wx.VERTICAL)
        
        self.user_input = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        input_sizer.Add(self.user_input, 1, wx.EXPAND | wx.ALL, 5)
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        send_btn = wx.Button(panel, label="Send")
        send_btn.Bind(wx.EVT_BUTTON, self.on_send_message)
        clear_btn = wx.Button(panel, label="Clear History")
        clear_btn.Bind(wx.EVT_BUTTON, self.on_clear_chat_history)
        
        btn_sizer.Add(send_btn, 1, wx.RIGHT, 5)
        btn_sizer.Add(clear_btn, 1)
        
        input_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(input_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        
    def create_settings_panel(self):
        """Create the settings panel."""
        panel = self.settings_panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # API key section
        api_key_box = wx.StaticBox(panel, label="API Key")
        api_key_sizer = wx.StaticBoxSizer(api_key_box, wx.VERTICAL)
        
        self.api_key_input = wx.TextCtrl(panel)
        self.api_key_input.SetValue(self.config_manager.get_api_key())
        api_key_sizer.Add(self.api_key_input, 0, wx.EXPAND | wx.ALL, 5)
        
        save_api_key_btn = wx.Button(panel, label="Save API Key")
        save_api_key_btn.Bind(wx.EVT_BUTTON, self.on_save_api_key)
        api_key_sizer.Add(save_api_key_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(api_key_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # PyAnnote token section
        pyannote_box = wx.StaticBox(panel, label="PyAnnote Speaker Diarization")
        pyannote_sizer = wx.StaticBoxSizer(pyannote_box, wx.VERTICAL)
        
        # Add a help text
        help_text = wx.StaticText(panel, 
                      label="To enable audio-based speaker identification, enter your Hugging Face token below:")
        pyannote_sizer.Add(help_text, 0, wx.EXPAND | wx.ALL, 5)
        
        self.pyannote_token_input = wx.TextCtrl(panel)
        self.pyannote_token_input.SetValue(self.config_manager.get_pyannote_token())
        pyannote_sizer.Add(self.pyannote_token_input, 0, wx.EXPAND | wx.ALL, 5)
        
        pyannote_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        save_pyannote_btn = wx.Button(panel, label="Save Token")
        save_pyannote_btn.Bind(wx.EVT_BUTTON, self.on_save_pyannote_token)
        pyannote_btn_sizer.Add(save_pyannote_btn, 1, wx.RIGHT, 5)
        
        get_token_btn = wx.Button(panel, label="Get Token Instructions")
        get_token_btn.Bind(wx.EVT_BUTTON, lambda e: self.show_pyannote_setup_guide())
        pyannote_btn_sizer.Add(get_token_btn, 1)
        
        pyannote_sizer.Add(pyannote_btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(pyannote_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Model selection section
        model_box = wx.StaticBox(panel, label="Model Selection")
        model_sizer = wx.StaticBoxSizer(model_box, wx.VERTICAL)
        
        self.model_choice = wx.Choice(panel, choices=["gpt-4o-mini", "gpt-3.5-turbo"])
        self.model_choice.SetSelection(0 if self.config_manager.get_model() == "gpt-4o-mini" else 1)
        model_sizer.Add(self.model_choice, 0, wx.EXPAND | wx.ALL, 5)
        
        save_model_btn = wx.Button(panel, label="Save Model")
        save_model_btn.Bind(wx.EVT_BUTTON, self.on_save_model)
        model_sizer.Add(save_model_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(model_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Temperature selection section
        temperature_box = wx.StaticBox(panel, label="Temperature")
        temperature_sizer = wx.StaticBoxSizer(temperature_box, wx.VERTICAL)
        
        self.temperature_slider = wx.Slider(panel, value=int(self.config_manager.get_temperature() * 10), minValue=0, maxValue=10)
        temperature_sizer.Add(self.temperature_slider, 0, wx.EXPAND | wx.ALL, 5)
        
        save_temperature_btn = wx.Button(panel, label="Save Temperature")
        save_temperature_btn.Bind(wx.EVT_BUTTON, self.on_save_temperature)
        temperature_sizer.Add(save_temperature_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(temperature_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Language selection section
        language_box = wx.StaticBox(panel, label="Language")
        language_sizer = wx.StaticBoxSizer(language_box, wx.VERTICAL)
        
        self.language_choice = wx.Choice(panel, choices=["English", "Hungarian"])
        self.language_choice.SetSelection(0 if self.config_manager.get_language() == "english" else 1)
        language_sizer.Add(self.language_choice, 0, wx.EXPAND | wx.ALL, 5)
        
        save_language_btn = wx.Button(panel, label="Save Language")
        save_language_btn.Bind(wx.EVT_BUTTON, self.on_save_language)
        language_sizer.Add(save_language_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(language_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Template management section
        template_box = wx.StaticBox(panel, label="Templates")
        template_sizer = wx.StaticBoxSizer(template_box, wx.VERTICAL)
        
        self.template_list = wx.ListBox(panel, style=wx.LB_SINGLE)
        template_sizer.Add(self.template_list, 1, wx.EXPAND | wx.ALL, 5)
        
        template_input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.template_name_input = wx.TextCtrl(panel)
        self.template_content_input = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        template_input_sizer.Add(self.template_name_input, 1, wx.EXPAND | wx.RIGHT, 5)
        template_input_sizer.Add(self.template_content_input, 2, wx.EXPAND)
        template_sizer.Add(template_input_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        template_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        add_template_btn = wx.Button(panel, label="Add Template")
        add_template_btn.Bind(wx.EVT_BUTTON, self.on_add_template)
        remove_template_btn = wx.Button(panel, label="Remove Template")
        remove_template_btn.Bind(wx.EVT_BUTTON, self.on_remove_template)
        template_btn_sizer.Add(add_template_btn, 1, wx.RIGHT, 5)
        template_btn_sizer.Add(remove_template_btn, 1)
        template_sizer.Add(template_btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(template_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        
        # Populate template list
        self.populate_template_list()
        
    def bind_events(self):
        """Bind events to handlers."""
        # Enter key in prompt input
        if hasattr(self, 'prompt_input'):
            self.prompt_input.Bind(wx.EVT_TEXT_ENTER, self.on_send_prompt)
        
    def on_close(self, event):
        """Handle application close event."""
        self.Destroy()
        
    def update_status(self, message, percent=None):
        """Update the status bar with a message and optional progress percentage."""
        self.status_bar.SetStatusText(message)
        
    def on_browse_audio(self, event):
        """Handle audio file browse button."""
        wildcard = (
            "Audio files|*.flac;*.m4a;*.mp3;*.mp4;*.mpeg;*.mpga;*.oga;*.ogg;*.wav;*.webm|"
            "FLAC files (*.flac)|*.flac|"
            "M4A files (*.m4a)|*.m4a|"
            "MP3 files (*.mp3)|*.mp3|"
            "MP4 files (*.mp4)|*.mp4|"
            "OGG files (*.ogg;*.oga)|*.ogg;*.oga|"
            "WAV files (*.wav)|*.wav|"
            "All files (*.*)|*.*"
        )
        
        with wx.FileDialog(self, "Choose an audio file", wildcard=wildcard,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
                
            path = file_dialog.GetPath()
            
            # Validate file extension
            file_ext = os.path.splitext(path)[1].lower()
            supported_formats = ['.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm']
            
            if file_ext not in supported_formats:
                # If user selected "All files" and chose an unsupported format
                wx.MessageBox(
                    f"The selected file has an unsupported format: {file_ext}\n"
                    f"Supported formats are: {', '.join(supported_formats)}", 
                    "Unsupported Format", 
                    wx.OK | wx.ICON_WARNING
                )
                return
                
            # Check file size
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            if file_size_mb > 25:
                wx.MessageBox(
                    f"The selected file is {file_size_mb:.1f}MB, which exceeds the 25MB limit for OpenAI's Whisper API.\n"
                    f"Please choose a smaller file or compress this one.",
                    "File Too Large",
                    wx.OK | wx.ICON_WARNING
                )
                return
                
            self.audio_file_path.SetValue(path)
            self.update_status(f"Selected audio file: {os.path.basename(path)} ({file_size_mb:.1f}MB)", percent=0)
            self.update_button_states()
            
    def on_transcribe(self, event):
        """Handle audio transcription."""
        if not self.audio_file_path.GetValue():
            wx.MessageBox("Please select an audio file first.", "No File Selected", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Check if API key is set
        if not self.config_manager.get_api_key():
            wx.MessageBox("Please set your OpenAI API key in the Settings tab.", "API Key Required", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Get language
        lang_map = {"English": "en", "Hungarian": "hu"}
        lang_selection = self.language_choice.GetString(self.language_choice.GetSelection())
        language = lang_map.get(lang_selection, "en")
        
        # Save language choice to config
        self.config_manager.set_language("english" if language == "en" else "hungarian")
        
        # Store the audio file path in the AudioProcessor
        # This ensures it's available for diarization later
        self.audio_processor.audio_file_path = self.audio_file_path.GetValue()
        
        # Update status message
        self.update_status(f"Transcribing in {lang_selection}...", percent=0)
        
        # Disable buttons during processing
        self.transcribe_btn.Disable()
        self.identify_speakers_btn.Disable()
        self.summarize_btn.Disable()
        
        # Start transcription in a separate thread
        threading.Thread(target=self.transcribe_thread, args=(self.audio_file_path.GetValue(), language)).start()
        
    def transcribe_thread(self, file_path, language):
        """Thread function for audio transcription."""
        try:
            # Get file extension for better error reporting
            file_ext = os.path.splitext(file_path)[1].lower()
            
            response = self.audio_processor.transcribe_audio(file_path, language)
            
            # Add a note about speaker identification at the top of the transcript
            transcription_notice = "--- TRANSCRIPTION COMPLETE ---\n" + \
                                  "To identify speakers in this transcript, click the 'Identify Speakers' button below.\n\n"
            
            wx.CallAfter(self.transcript_text.SetValue, transcription_notice + self.audio_processor.transcript)
            wx.CallAfter(self.update_button_states)
            wx.CallAfter(self.update_status, f"Transcription complete: {len(self.audio_processor.transcript)} characters", percent=100)
            
            # Show a dialog informing the user to use speaker identification
            wx.CallAfter(self.show_speaker_id_hint)
            
        except FileNotFoundError as e:
            wx.CallAfter(wx.MessageBox, f"File not found: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
        except ValueError as e:
            error_msg = str(e)
            title = "Format Error"
            
            # Special handling for common error cases
            if 'ffprobe' in error_msg or 'ffmpeg' in error_msg:
                title = "FFmpeg Missing"
                error_msg = error_msg.replace('[Errno 2] No such file or directory:', 'Missing required component:')
                # Installation instructions are already in the error message from _get_ffmpeg_install_instructions
            elif file_ext == '.m4a' and 'Invalid file format' in error_msg:
                error_msg = (
                    "There was an issue with your M4A file. Some M4A files have compatibility issues with the OpenAI API.\n\n"
                    "Possible solutions:\n"
                    "1. Install FFmpeg on your system (required for m4a processing)\n"
                    "2. Convert the file to WAV or MP3 format manually\n"
                    "3. Try a different M4A file (some are more compatible than others)"
                )
                title = "M4A Compatibility Issue"
                
            wx.CallAfter(wx.MessageBox, error_msg, title, wx.OK | wx.ICON_ERROR)
        except openai.RateLimitError:
            wx.CallAfter(wx.MessageBox, "OpenAI rate limit exceeded. Please try again later.", "Rate Limit Error", wx.OK | wx.ICON_ERROR)
        except openai.AuthenticationError:
            wx.CallAfter(wx.MessageBox, "Authentication error. Please check your OpenAI API key in the Settings tab.", "Authentication Error", wx.OK | wx.ICON_ERROR)
        except openai.BadRequestError as e:
            error_msg = str(e)
            title = "API Error"
            
            if "Invalid file format" in error_msg and file_ext == '.m4a':
                error_msg = (
                    "Your M4A file format is not compatible with the OpenAI API.\n\n"
                    "Possible solutions:\n"
                    "1. Install FFmpeg on your system (required for m4a processing)\n"
                    "2. Convert the file to WAV or MP3 format manually\n"
                    "3. Try a different M4A file (some are more compatible than others)"
                )
                title = "M4A Format Error"
                
            wx.CallAfter(wx.MessageBox, error_msg, title, wx.OK | wx.ICON_ERROR)
        except Exception as e:
            error_msg = str(e)
            if 'ffprobe' in error_msg or 'ffmpeg' in error_msg:
                # Handle FFmpeg-related errors not caught by previous handlers
                install_instructions = self.audio_processor._get_ffmpeg_install_instructions()
                error_msg = f"FFmpeg/FFprobe is required but not found. Please install it to process audio files.\n\n{install_instructions}"
                wx.CallAfter(wx.MessageBox, error_msg, "FFmpeg Required", wx.OK | wx.ICON_ERROR)
            else:
                wx.CallAfter(wx.MessageBox, f"Transcription error: {error_msg}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            wx.CallAfter(self.transcribe_btn.Enable)
            wx.CallAfter(self.update_status, "Ready", percent=0)
            
    def show_speaker_id_hint(self):
        """Show a hint dialog about using speaker identification."""
        # Check if PyAnnote is available
        if PYANNOTE_AVAILABLE:
            message = (
                "Transcription is complete!\n\n"
                "To identify different speakers in this transcript, click the 'Identify Speakers' button.\n\n"
                "This system will use advanced audio-based speaker diarization to detect different "
                "speakers by analyzing voice characteristics (pitch, tone, speaking style) from the "
                "original audio file.\n\n"
                "This approach is significantly more accurate than text-based analysis since it "
                "uses the actual voice patterns to distinguish between speakers."
            )
        else:
            message = (
                "Transcription is complete!\n\n"
                "To identify different speakers in this transcript, click the 'Identify Speakers' button.\n\n"
                "Currently, the system will analyze the text patterns to detect different speakers.\n\n"
                "For more accurate speaker identification, consider installing PyAnnote which uses "
                "audio analysis to distinguish speakers based on their voice characteristics. "
                "Click 'Yes' for installation instructions."
            )
            
        dlg = wx.MessageDialog(
            self,
            message,
            "Speaker Identification",
            wx.OK | (wx.CANCEL | wx.YES_NO if not PYANNOTE_AVAILABLE else wx.OK) | wx.ICON_INFORMATION
        )
        
        result = dlg.ShowModal()
        dlg.Destroy()
        
        # If user wants to install PyAnnote
        if result == wx.ID_YES:
            self.show_pyannote_setup_guide()
        
        # Highlight the identify speakers button
        self.identify_speakers_btn.SetFocus()
        
    def on_identify_speakers(self, event):
        """Handle speaker identification."""
        if not self.audio_processor.transcript:
            wx.MessageBox("Please transcribe an audio file first.", "No Transcript", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Create and show progress dialog
        progress_dialog = wx.ProgressDialog(
            "Speaker Identification",
            "Processing speaker identification...\n\nAnalyzing transcript for different speakers...",
            maximum=100,
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_ELAPSED_TIME
        )
        progress_dialog.Update(20)  # Initial update
        
        # Disable buttons during processing
        self.identify_speakers_btn.Disable()
        
        # Start speaker identification in a separate thread
        threading.Thread(target=self.identify_speakers_thread, args=(progress_dialog,)).start()
    
    def identify_speakers_thread(self, progress_dialog=None):
        """Run speaker identification in a separate thread."""
        try:
            # Get transcript from the transcript output field
            transcript = self.transcript_text.GetValue()
            
            # Check if transcript exists
            if not transcript:
                raise ValueError("No transcript available. Please transcribe an audio file first.")
            
            # Create progress dialog if not provided
            if not progress_dialog:
                progress_dialog = wx.ProgressDialog(
                    "Identifying Speakers",
                    "Initializing speaker identification...",
                    maximum=100,
                    parent=self,
                    style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_SMOOTH | wx.PD_CAN_ABORT
                )
            
            # Register update callback for status updates
            def on_progress_update(message, percent=None):
                if percent is not None:
                    wx.CallAfter(progress_dialog.Update, int(percent * 100), message)
                else:
                    wx.CallAfter(progress_dialog.Pulse, message)
            
            # First check if we have audio file for diarization
            file_path = self.audio_file_path.GetValue() if hasattr(self, 'audio_file_path') else None
            
            if file_path and os.path.exists(file_path) and PYANNOTE_AVAILABLE:
                # Let user know we're doing audio-based speaker ID
                wx.CallAfter(progress_dialog.Update, 5, "Initializing audio diarization...")
                
                # Call identify_speakers_with_diarization
                speakers = self.audio_processor.identify_speakers_with_diarization(file_path, transcript)
            else:
                # Fall back to text-based approach
                wx.CallAfter(progress_dialog.Update, 5, "Using text-based speaker identification...")
                speakers = self.audio_processor.identify_speakers_simple(transcript)
            
            # Check the result
            if not speakers or len(speakers) == 0:
                wx.CallAfter(progress_dialog.Update, 100, "Speaker identification failed.")
                wx.CallAfter(wx.MessageBox, "Unable to identify speakers.", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Close progress dialog
            wx.CallAfter(progress_dialog.Update, 100, "Speaker identification complete!")
            
            # Create the speaker mapping UI
            wx.CallAfter(self.create_speaker_mapping_ui, speakers)
            
            # Update transcript with speaker labels for preview
            formatted_transcript = self.format_transcript_with_speakers(speakers)
            wx.CallAfter(self.transcript_text.SetValue, formatted_transcript)
            
            # Enable Apply button
            wx.CallAfter(self.update_button_states)
            
        except Exception as e:
            wx.CallAfter(progress_dialog.Update, 100, f"Error: {str(e)}")
            wx.CallAfter(wx.MessageBox, f"Error identifying speakers: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
            import traceback
            traceback.print_exc()
    
    def format_transcript_with_speakers(self, speakers):
        """Format transcript with speaker labels for display purposes."""
        if not speakers:
            return self.audio_processor.transcript
            
        formatted_text = []
        
        for segment in speakers:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "")
            
            # Format with bold speaker name
            formatted_text.append(f"{speaker}: {text}")
            
        return "\n\n".join(formatted_text)
    
    def show_token_missing_dialog(self):
        """Show a dialog explaining that PyAnnote is installed but the token is missing."""
        dlg = wx.MessageDialog(
            self,
            "PyAnnote is installed, but no token has been configured.\n\n"
            "Audio-based speaker identification requires a HuggingFace token.\n\n"
            "Would you like to go to the Settings tab to add your token?",
            "PyAnnote Token Missing",
            wx.YES_NO | wx.ICON_INFORMATION
        )
        
        if dlg.ShowModal() == wx.ID_YES:
            self.notebook.SetSelection(2)  # Switch to settings tab
            # Add focus to the token input field
            self.pyannote_token_input.SetFocus()
        else:
            # Continue with text-based identification
            threading.Thread(target=self.continue_with_text_identification).start()
            
        dlg.Destroy()
        
    def continue_with_text_identification(self):
        """Continue with text-based speaker identification after token missing dialog."""
        progress_dialog = wx.ProgressDialog(
            "Speaker Identification",
            "Proceeding with text-based analysis...",
            maximum=100,
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_ELAPSED_TIME
        )
        progress_dialog.Update(30)
        
        try:
            # Force text-based identification regardless of PyAnnote availability
            speakers = self.audio_processor.identify_speakers_simple(self.audio_processor.transcript)
            
            wx.CallAfter(progress_dialog.Update, 80, "Creating speaker mapping interface...")
            
            # Clear existing mapping UI
            wx.CallAfter(self.speaker_mapping_sizer.Clear, True)
            
            # Create UI for speaker mapping
            if speakers:
                wx.CallAfter(self.create_speaker_mapping_ui, speakers)
                speaker_count = len(set(s["speaker"] for s in speakers))
                
                wx.CallAfter(self.update_status, 
                            f"Speaker identification complete using text analysis. Found {speaker_count} speakers.", percent=100)
                
                wx.CallAfter(progress_dialog.Update, 100, 
                            f"Found {speaker_count} speakers using text analysis!")
            else:
                wx.CallAfter(self.update_status, "No speakers identified.", percent=100)
                wx.CallAfter(progress_dialog.Update, 100, "No speakers were identified in the transcript.")
                
        except Exception as e:
            wx.CallAfter(progress_dialog.Destroy)
            wx.CallAfter(wx.MessageBox, f"Speaker identification error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            wx.CallAfter(self.identify_speakers_btn.Enable)

    def create_speaker_mapping_ui(self, speakers):
        """Create UI for speaker name mapping."""
        speaker_ids = set()
        for segment in speakers:
            speaker_ids.add(segment["speaker"])
            
        self.speaker_inputs = {}
        
        # Use generic naming as default as these work better for unidentified speakers
        # Start with Speaker 1, Speaker 2, etc.
        
        i = 0
        for speaker_id in sorted(speaker_ids):
            label = wx.StaticText(self.speaker_mapping_panel, label=f"{speaker_id}:")
            text_input = wx.TextCtrl(self.speaker_mapping_panel)
            
            # Keep the existing speaker ID if it follows our naming convention (Speaker X)
            # or if it already appears to be a proper name
            if "Speaker" in speaker_id or any(char.isupper() for char in speaker_id[1:]):
                text_input.SetValue(speaker_id)
            else:
                # Otherwise assign a generic Speaker number
                text_input.SetValue(f"Speaker {i+1}")
            
            self.speaker_mapping_sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL)
            self.speaker_mapping_sizer.Add(text_input, 1, wx.EXPAND)
            
            self.speaker_inputs[speaker_id] = text_input
            i += 1
            
        # Add a label with instructions
        help_text = wx.StaticText(self.speaker_mapping_panel, 
                                 label="Customize the speaker names above and then click 'Apply Speaker Names'")
        # Span both columns
        self.speaker_mapping_sizer.Add(help_text, 0, wx.ALIGN_CENTER | wx.TOP, 10)
        self.speaker_mapping_sizer.Add(wx.StaticText(self.speaker_mapping_panel, label=""), 0)
            
        self.speaker_mapping_panel.Layout()
        self.audio_panel.Layout()
            
        # Auto-apply the speaker names to give immediate feedback
        wx.CallLater(500, self.on_apply_speaker_names, None)
        
    def on_apply_speaker_names(self, event):
        """Apply speaker names to the transcript."""
        if not hasattr(self, 'speaker_inputs') or not self.speaker_inputs:
            if event is not None:  # Only show message if called directly
                wx.MessageBox("Please identify speakers first.", "No Speakers Identified", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Get speaker name mapping
        speaker_map = {sid: input_ctrl.GetValue() for sid, input_ctrl in self.speaker_inputs.items()}
        
        # Apply speaker names
        updated_transcript = self.audio_processor.assign_speaker_names(speaker_map)
        
        # Update transcript display
        self.transcript_text.SetValue("")  # Clear first to reset styling
        
        # Check which method was used for speaker identification
        using_diarization = hasattr(self.audio_processor, 'diarization') and self.audio_processor.diarization
        
        # Add a header indicating which method was used
        if using_diarization:
            method_text = "Speaker identification performed using audio voice analysis (PyAnnote)"
            method_color = wx.Colour(0, 128, 0)  # Green for audio-based
        else:
            method_text = "Speaker identification performed using text pattern analysis"
            method_color = wx.Colour(128, 0, 0)  # Red for text-based
            
        # Add the method header with appropriate styling
        self.transcript_text.SetDefaultStyle(wx.TextAttr(method_color, wx.NullColour, 
                                            wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL)))
        self.transcript_text.AppendText(method_text + "\n\n")
        
        # Add each speaker segment with styling
        lines = updated_transcript.split("\n\n")
        for i, line in enumerate(lines):
            if ":" in line:
                speaker, text = line.split(":", 1)
                
                # Add speaker name with bold style and larger font
                speaker_font = wx.Font(wx.NORMAL_FONT.GetPointSize() + 1, wx.FONTFAMILY_DEFAULT, 
                                     wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
                self.transcript_text.SetDefaultStyle(wx.TextAttr(wx.BLUE, wx.NullColour, speaker_font))
                self.transcript_text.AppendText(f"{speaker}:")
                
                # Add text with normal style
                self.transcript_text.SetDefaultStyle(wx.TextAttr(wx.BLACK, wx.NullColour, wx.Font(wx.NORMAL_FONT.GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)))
                self.transcript_text.AppendText(f"{text}")
            else:
                self.transcript_text.AppendText(line)
                
            # Add newlines between segments (except for the last one)
            if i < len(lines) - 1:
                self.transcript_text.AppendText("\n\n")
                
        if event is not None:  # Only show status message if called directly
            method_description = "audio diarization" if using_diarization else "text analysis"
            self.update_status(f"Speaker names applied to transcript using {method_description}.", percent=100)
        
    def on_summarize(self, event):
        """Generate a summary of the transcript."""
        if not self.audio_processor.transcript:
            wx.MessageBox("Please transcribe an audio file first.", "No Transcript", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Get selected template
        template_idx = self.template_choice.GetSelection()
        template_name = None
        if template_idx > 0:  # 0 is "None"
            template_name = self.template_choice.GetString(template_idx)
            
        # Disable button during processing
        self.summarize_btn.Disable()
        
        # Start summarization in a separate thread
        transcript = self.transcript_text.GetValue()
        threading.Thread(target=self.summarize_thread, args=(transcript, template_name)).start()
        
    def summarize_thread(self, transcript, template_name):
        """Thread function for transcript summarization."""
        try:
            summary = self.llm_processor.summarize_transcript(transcript, template_name)
            
            # Show summary in a dialog
            wx.CallAfter(self.show_summary_dialog, summary)
        except Exception as e:
            wx.CallAfter(wx.MessageBox, f"Summarization error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            wx.CallAfter(self.summarize_btn.Enable)
            
    def show_summary_dialog(self, summary):
        """Show summary in a dialog."""
        dlg = wx.Dialog(self, title="Summary", size=(600, 400))
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        text_ctrl = wx.TextCtrl(dlg, style=wx.TE_MULTILINE | wx.TE_READONLY)
        text_ctrl.SetValue(summary)
        
        sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALL, 10)
        
        # Add Close button
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        close_btn = wx.Button(dlg, wx.ID_CLOSE)
        btn_sizer.Add(close_btn, 0, wx.ALL, 5)
        
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        dlg.SetSizer(sizer)
        
        close_btn.Bind(wx.EVT_BUTTON, lambda event: dlg.EndModal(wx.ID_CLOSE))
        
        dlg.ShowModal()
        dlg.Destroy()
        
    def update_button_states(self):
        """Update the enabled/disabled states of buttons based on current state."""
        has_audio_file = bool(self.audio_file_path.GetValue())
        has_transcript = hasattr(self.audio_processor, 'transcript') and bool(self.audio_processor.transcript)
        has_speakers = hasattr(self.audio_processor, 'speakers') and bool(self.audio_processor.speakers)
        
        if hasattr(self, 'transcribe_btn'):
            self.transcribe_btn.Enable(has_audio_file)
            
        if hasattr(self, 'identify_speakers_btn'):
            self.identify_speakers_btn.Enable(has_transcript)
            
        if hasattr(self, 'apply_speaker_names_btn'):
            self.apply_speaker_names_btn.Enable(has_speakers)
            
        if hasattr(self, 'summarize_btn'):
            self.summarize_btn.Enable(has_transcript)
        
    def on_send_message(self, event):
        """Handle sending a message in the chat."""
        user_input = self.user_input.GetValue()
        if not user_input:
            return
            
        # Generate response
        response = self.llm_processor.generate_response(user_input)
        
        # Update chat history
        self.chat_history_text.AppendText(f"You: {user_input}\n")
        self.chat_history_text.AppendText(f"Assistant: {response}\n\n")
        
        # Clear user input
        self.user_input.SetValue("")
        
    def on_clear_chat_history(self, event):
        """Clear the chat history."""
        self.llm_processor.clear_chat_history()
        self.chat_history_text.SetValue("")
        
    def on_save_api_key(self, event):
        """Save the API key."""
        api_key = self.api_key_input.GetValue()
        self.config_manager.set_api_key(api_key)
        wx.MessageBox("API key saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def on_save_model(self, event):
        """Save the selected model."""
        model = self.model_choice.GetString(self.model_choice.GetSelection())
        self.config_manager.set_model(model)
        wx.MessageBox("Model saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def on_save_temperature(self, event):
        """Save the temperature value."""
        temperature = self.temperature_slider.GetValue() / 10.0
        self.config_manager.set_temperature(temperature)
        wx.MessageBox("Temperature saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def on_save_language(self, event):
        """Save the selected language."""
        language = self.language_choice.GetString(self.language_choice.GetSelection()).lower()
        self.config_manager.set_language(language)
        wx.MessageBox("Language saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def populate_template_list(self):
        """Populate the template list with available templates."""
        self.template_list.Clear()
        templates = self.config_manager.get_templates()
        for name in templates.keys():
            self.template_list.Append(name)
            
    def on_add_template(self, event):
        """Add a new template."""
        name = self.template_name_input.GetValue()
        content = self.template_content_input.GetValue()
        
        if not name or not content:
            wx.MessageBox("Please enter both name and content for the template.", "Error", wx.OK | wx.ICON_ERROR)
            return
            
        self.config_manager.add_template(name, content)
        self.populate_template_list()
        self.template_name_input.SetValue("")
        self.template_content_input.SetValue("")
        
    def on_remove_template(self, event):
        if template_name in templates:
            self.template_content_input.SetValue(templates[template_name])
        else:
            self.template_content_input.Clear()
            
    def on_new_template(self, event):
        """Create a new template."""
        dlg = wx.TextEntryDialog(self, "Enter template name:", "New Template")
        if dlg.ShowModal() == wx.ID_OK:
            template_name = dlg.GetValue()
            
            if not template_name:
                wx.MessageBox("Template name cannot be empty.", "Invalid Name", wx.OK | wx.ICON_ERROR)
                return
                
            templates = self.config_manager.get_templates()
            if template_name in templates:
                wx.MessageBox(f"Template '{template_name}' already exists.", "Duplicate Name", wx.OK | wx.ICON_ERROR)
                return
                
            # Add new template
            self.config_manager.add_template(template_name, "")
            
            # Update template choice
            templates = list(self.config_manager.get_templates().keys())
            self.template_choice.SetItems(templates)
            self.template_choice.SetSelection(templates.index(template_name))
            
            # Clear content
            self.template_content_input.Clear()
            
        dlg.Destroy()
        
    def on_save_template(self, event):
        """Save the current template."""
        templates = list(self.config_manager.get_templates().keys())
        if not templates:
            wx.MessageBox("No templates available. Create a new template first.", "No Templates", wx.OK | wx.ICON_INFORMATION)
            return
            
        template_name = self.template_choice.GetString(self.template_choice.GetSelection())
        template_content = self.template_content_input.GetValue()
        
        # Update template
        self.config_manager.add_template(template_name, template_content)
        
        wx.MessageBox(f"Template '{template_name}' saved.", "Template Saved", wx.OK | wx.ICON_INFORMATION)
        
    def on_delete_template(self, event):
        """Delete the selected template."""
        templates = list(self.config_manager.get_templates().keys())
        if not templates:
            return
            
        template_name = self.template_choice.GetString(self.template_choice.GetSelection())
        
        # Confirm deletion
        dlg = wx.MessageDialog(self, f"Are you sure you want to delete the template '{template_name}'?",
                              "Confirm Deletion", wx.YES_NO | wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_YES:
            # Delete template
            self.config_manager.remove_template(template_name)
            
            # Update template choice
            templates = list(self.config_manager.get_templates().keys())
            self.template_choice.SetItems(templates if templates else ["No templates"])
            self.template_choice.SetSelection(0)
            
            # Clear content
            self.template_content_input.Clear()
            
            if templates:
                self.load_template(templates[0])
                
        dlg.Destroy()

    def show_format_info(self):
        """Show information about supported audio formats."""
        ffmpeg_missing = not self._is_ffmpeg_available()
        pydub_missing = not PYDUB_AVAILABLE
        
        if ffmpeg_missing or pydub_missing:
            needed_tools = []
            if pydub_missing:
                needed_tools.append("pydub (pip install pydub)")
            if ffmpeg_missing:
                needed_tools.append("FFmpeg")
                
            # Get platform-specific installation instructions
            ffmpeg_install = self.audio_processor._get_ffmpeg_install_instructions() if hasattr(self, 'audio_processor') else ""
            
            msg = (
                "For better audio file compatibility, especially with M4A files, "
                f"you need to install the following tools:\n\n{', '.join(needed_tools)}\n\n"
            )
            
            if ffmpeg_missing:
                msg += f"FFmpeg installation instructions:\n{ffmpeg_install}\n\n"
                msg += "FFmpeg is required for processing M4A files. Without it, M4A transcription will likely fail."
            
            self.update_status("FFmpeg required for M4A support - please install it", percent=0)
            
            # Always show FFmpeg warning because it's critical
            if ffmpeg_missing:
                wx.MessageBox(msg, "FFmpeg Required for M4A Files", wx.OK | wx.ICON_WARNING)
                self.config_manager.config["shown_format_info"] = True
                self.config_manager.save_config()
            # Only show other warnings if not shown before
            elif not self.config_manager.config.get("shown_format_info", False):
                wx.MessageBox(msg, "Audio Format Information", wx.OK | wx.ICON_INFORMATION)
                self.config_manager.config["shown_format_info"] = True
                self.config_manager.save_config()

    def _is_ffmpeg_available(self):
        """Check if ffmpeg is available on the system."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def check_pyannote(self):
        """Check if PyAnnote is available and show installation instructions if not."""
        if not PYANNOTE_AVAILABLE:
            dlg = wx.MessageDialog(
                self,
                "PyAnnote is not installed. PyAnnote provides more accurate speaker diarization "
                "by analyzing audio directly, rather than just text.\n\n"
                "To install PyAnnote and set it up, click 'Yes' for detailed instructions.",
                "Speaker Diarization Enhancement",
                wx.YES_NO | wx.ICON_INFORMATION
            )
            if dlg.ShowModal() == wx.ID_YES:
                self.show_pyannote_setup_guide()
            dlg.Destroy()
    
    def show_pyannote_setup_guide(self):
        """Show detailed setup instructions for PyAnnote."""
        dlg = wx.Dialog(self, title="PyAnnote Setup Guide", size=(650, 550))
        
        panel = wx.Panel(dlg)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create a styled text control for better formatting
        text = wx.TextCtrl(
            panel, 
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
            size=(-1, 400)
        )
        
        # Set up the instructions
        guide = """PYANNOTE SETUP GUIDE

Step 1: Install Required Dependencies
--------------------------------------
Run the following commands in your terminal:

pip install torch torchaudio
pip install pyannote.audio

Step 2: Get HuggingFace Access Token
------------------------------------
1. Create a HuggingFace account at https://huggingface.co/join
2. Go to https://huggingface.co/pyannote/speaker-diarization
3. Accept the user agreement
4. Go to https://huggingface.co/settings/tokens
5. Create a new token with READ access
6. Copy the token

Step 3: Configure the Application
--------------------------------
1. After installing, restart this application
2. Go to the Settings tab
3. Paste your token in the "PyAnnote Speaker Diarization" section
4. Click "Save Token"
5. Return to the Audio Processing tab
6. Click "Identify Speakers" to use audio-based speaker identification

Important Notes:
---------------
• PyAnnote requires at least 4GB of RAM
• GPU acceleration (if available) will make processing much faster
• For best results, use high-quality audio with minimal background noise
• The first run may take longer as models are downloaded

Troubleshooting:
---------------
• If you get CUDA errors, try installing a compatible PyTorch version for your GPU
• If you get "Access Denied" errors, check that your token is valid and you've accepted the license agreement
• For long audio files (>10 min), processing may take several minutes
"""
        
        # Add the text with some styling
        text.SetValue(guide)
        
        # Style the headers
        text.SetStyle(0, 19, wx.TextAttr(wx.BLUE, wx.NullColour, wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)))
        
        # Find all the section headers and style them
        for section in ["Step 1:", "Step 2:", "Step 3:", "Important Notes:", "Troubleshooting:"]:
            start = guide.find(section)
            if start != -1:
                end = start + len(section)
                text.SetStyle(start, end, wx.TextAttr(wx.Colour(128, 0, 128), wx.NullColour, wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)))
        
        # Add to sizer
        sizer.Add(text, 1, wx.EXPAND | wx.ALL, 10)
        
        # Add buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Add a button to copy installation commands
        copy_btn = wx.Button(panel, label="Copy Installation Commands")
        copy_btn.Bind(wx.EVT_BUTTON, lambda e: self.copy_to_clipboard("pip install torch torchaudio\npip install pyannote.audio"))
        btn_sizer.Add(copy_btn, 0, wx.RIGHT, 10)
        
        # Add a button to open HuggingFace token page
        hf_btn = wx.Button(panel, label="Open HuggingFace Token Page")
        hf_btn.Bind(wx.EVT_BUTTON, lambda e: wx.LaunchDefaultBrowser("https://huggingface.co/settings/tokens"))
        btn_sizer.Add(hf_btn, 0, wx.RIGHT, 10)
        
        # Add button to go to settings tab
        settings_btn = wx.Button(panel, label="Go to Settings Tab")
        settings_btn.Bind(wx.EVT_BUTTON, lambda e: (self.notebook.SetSelection(2), dlg.EndModal(wx.ID_CLOSE)))
        btn_sizer.Add(settings_btn, 0, wx.RIGHT, 10)
        
        # Add close button
        close_btn = wx.Button(panel, wx.ID_CLOSE)
        close_btn.Bind(wx.EVT_BUTTON, lambda e: dlg.EndModal(wx.ID_CLOSE))
        btn_sizer.Add(close_btn, 0)
        
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        dlg.ShowModal()
        dlg.Destroy()
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard."""
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(text))
            wx.TheClipboard.Close()
            wx.MessageBox("Commands copied to clipboard", "Copied", wx.OK | wx.ICON_INFORMATION)

    def on_save_pyannote_token(self, event):
        """Save the PyAnnote token."""
        token = self.pyannote_token_input.GetValue()
        self.config_manager.set_pyannote_token(token)
        wx.MessageBox("PyAnnote token saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)

# Main Application Class
class AudioApp(wx.App):
    def OnInit(self):
        frame = MainFrame(None, app_name)
        frame.Show()
        return True

# Main function
if __name__ == "__main__":
    # Ensure required directories exist
    ensure_directories()
    
    # Create and start the application
    app = AudioApp()
    app.MainLoop()