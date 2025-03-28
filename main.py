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

# Check if pydub is available for audio conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

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

# Audio Processing Class
class AudioProcessor:
    def __init__(self, client, update_callback=None):
        self.client = client
        self.update_callback = update_callback
        self.transcript = ""
        self.speakers = []
        self.word_by_word = []
        
    def update_status(self, message):
        if self.update_callback:
            wx.CallAfter(self.update_callback, message)
            
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
        """Convert audio file to WAV format for better compatibility."""
        self.update_status(f"Converting {os.path.basename(file_path)} to WAV format...")
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # If already WAV, just return the original file
        if file_ext == '.wav':
            return file_path
            
        # Create a temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        try:
            if PYDUB_AVAILABLE:
                try:
                    # Use pydub for conversion
                    if file_ext == '.mp3':
                        audio = AudioSegment.from_mp3(file_path)
                    elif file_ext == '.m4a':
                        audio = AudioSegment.from_file(file_path, format="m4a")
                    else:
                        audio = AudioSegment.from_file(file_path)
                    
                    audio.export(temp_wav.name, format="wav")
                    self.update_status("Conversion complete using pydub.")
                    return temp_wav.name
                except FileNotFoundError as e:
                    # This usually means ffmpeg/ffprobe is not installed or not in PATH
                    if "ffprobe" in str(e) or "ffmpeg" in str(e):
                        self.update_status("Pydub requires FFmpeg to be installed. Trying direct FFmpeg...")
                        # Fall through to the FFmpeg method
                    else:
                        raise
            
            # Fallback to direct ffmpeg if installed
            try:
                # Check if ffmpeg is available
                subprocess.run(["ffmpeg", "-version"], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=True)
                
                # Convert using ffmpeg
                self.update_status("Converting with FFmpeg...")
                subprocess.run([
                    "ffmpeg", 
                    "-i", file_path, 
                    "-ar", "16000",  # Whisper works best with 16kHz
                    "-ac", "1",      # Mono channel
                    "-c:a", "pcm_s16le",  # 16-bit PCM
                    "-y",            # Overwrite output file
                    temp_wav.name
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                self.update_status("Conversion complete using FFmpeg.")
                return temp_wav.name
                
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                # If ffmpeg failed or is not available
                install_instructions = self._get_ffmpeg_install_instructions()
                error_msg = (
                    f"Could not convert audio file: FFmpeg/FFprobe is not installed or not in your PATH. "
                    f"\n\nTo install FFmpeg: {install_instructions}"
                )
                raise ValueError(error_msg)
                    
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
            raise ValueError(f"Error converting audio file: {str(e)}")
    
    def _get_ffmpeg_install_instructions(self):
        """Return platform-specific instructions for installing FFmpeg."""
        if sys.platform == 'darwin':  # macOS
            return "brew install ffmpeg  (using Homebrew) or visit https://ffmpeg.org/download.html"
        elif sys.platform == 'win32':  # Windows
            return "Download from https://ffmpeg.org/download.html or install using Chocolatey: choco install ffmpeg"
        else:  # Linux
            return "sudo apt install ffmpeg  (Debian/Ubuntu) or sudo yum install ffmpeg (Fedora/CentOS)"
            
    def transcribe_audio(self, file_path, language="en"):
        """Transcribe audio file using OpenAI Whisper API."""
        converted_file = None
        
        try:
            # Validate the audio file
            self.validate_audio_file(file_path)
            
            # Get file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.update_status(f"Processing {os.path.basename(file_path)} ({file_ext} format, {file_size_mb:.2f}MB)")
            
            # For m4a files, we'll directly pass the file without using temporary files
            # as the API supports m4a natively
            self.update_status(f"Transcribing audio file: {os.path.basename(file_path)}")
            
            # Open the file directly - no need for temporary files or conversion for supported formats
            try:
                with open(file_path, "rb") as audio_file:
                    self.update_status("Sending file to OpenAI Whisper API...")
                    response = self.client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                        response_format="verbose_json",
                        language=language,
                        timestamp_granularities=["word"]
                    )
            except openai.BadRequestError as e:
                if "Invalid file format" in str(e) and file_ext == '.m4a':
                    # Special handling for m4a files that sometimes have compatibility issues
                    self.update_status("M4A format issue detected. Trying with conversion...")
                    
                    # Try to convert to wav as a fallback
                    if PYDUB_AVAILABLE or self._is_ffmpeg_available():
                        converted_file = self.convert_to_wav(file_path)
                        with open(converted_file, "rb") as audio_file:
                            self.update_status("Sending converted file to OpenAI Whisper API...")
                            response = self.client.audio.transcriptions.create(
                                model=WHISPER_MODEL,
                                file=audio_file,
                                response_format="verbose_json",
                                language=language,
                                timestamp_granularities=["word"]
                            )
                    else:
                        # If conversion tools aren't available, raise a more helpful error
                        raise ValueError(
                            "This M4A file has compatibility issues with the OpenAI API. "
                            "Please install pydub (pip install pydub) or ffmpeg for automatic conversion, "
                            "or convert it manually to WAV format before uploading."
                        )
                else:
                    # Re-raise if it's not an m4a format issue
                    raise
                
            self.transcript = response.text
            self.word_by_word = response.words
            
            # Save raw transcript
            transcript_filename = f"Transcripts/transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(transcript_filename, 'w') as f:
                f.write(json.dumps(response.model_dump(), indent=2))
                
            self.update_status("Transcription complete.")
            return response
            
        except openai.APIError as e:
            error_msg = f"OpenAI API Error: {str(e)}"
            self.update_status(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            self.update_status(error_msg)
            raise
        finally:
            # Clean up converted file if it was created
            if converted_file and os.path.exists(converted_file):
                try:
                    os.unlink(converted_file)
                except:
                    pass
                    
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
        """Use OpenAI to identify different speakers in the transcript."""
        self.update_status("Identifying speakers...")
        
        # Use gpt-4o-mini for speaker identification
        model_to_use = DEFAULT_OPENAI_MODEL  # gpt-4o-mini is set as the default model
        
        self.update_status(f"Using {model_to_use} for speaker identification...")
        
        # Split the transcript into segments that might be from different speakers
        lines = transcript.split('\n')
        segments = []
        current_segment = []
        
        for line in lines:
            if line.strip():
                current_segment.append(line)
            elif current_segment:
                segments.append(' '.join(current_segment))
                current_segment = []
                
        if current_segment:
            segments.append(' '.join(current_segment))
        
        # If transcript doesn't have clear segments, create artificial breaks
        if len(segments) < 2:
            words = transcript.split()
            segment_size = max(20, len(words) // max(2, len(words) // 100))
            segments = []
            
            for i in range(0, len(words), segment_size):
                segments.append(' '.join(words[i:i+segment_size]))
        
        # Now create the prompt with these segments
        prompt = f"""
        Your task is to identify different speakers in this transcript.
        
        IMPORTANT INSTRUCTIONS:
        1. You MUST assign each segment to a speaker, even if you're uncertain.
        2. You MUST identify at least 2 different speakers if the transcript isn't clearly a monologue.
        3. Always use "Speaker 1", "Speaker 2", etc. as identifiers.
        4. Analyze turn-taking patterns, question-answer sequences, and different speaking styles.
        5. Look for language patterns that indicate different people.
        
        Format your response exactly as a JSON array with this structure:
        [
            {{"speaker": "Speaker 1", "text": "Speaker 1's text..."}},
            {{"speaker": "Speaker 2", "text": "Speaker 2's text..."}},
            ...
        ]
        
        Here's the transcript split into possible segments:

        {json.dumps(segments, indent=2)}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are a transcript analyzer that must divide any conversational text into parts spoken by different speakers. You always assume multiple speakers unless it's obviously a single-person monologue."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            self.speakers = result if isinstance(result, list) else result.get("speakers", [])
            
            # Ensure we have at least some speakers identified
            if not self.speakers:
                # Fallback: create basic speaker segmentation
                self.speakers = self._create_default_speaker_segments(transcript)
            
            self.update_status(f"Speaker identification complete. Found {len(set(s['speaker'] for s in self.speakers))} speakers.")
            return self.speakers
            
        except Exception as e:
            self.update_status(f"Error identifying speakers: {str(e)}")
            # Fallback in case of API error
            self.speakers = self._create_default_speaker_segments(transcript)
            return self.speakers
            
    def _create_default_speaker_segments(self, transcript):
        """Create fallback speaker segments when API fails."""
        # Split the transcript into sentences or paragraphs
        import re
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        speakers = []
        current_speaker = "Speaker 1"
        
        # Assign alternating speakers to sentences
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # Switch speakers every few sentences
            if i > 0 and i % 3 == 0:
                current_speaker = f"Speaker {(i // 3) % 2 + 1}"
                
            speakers.append({
                "speaker": current_speaker,
                "text": sentence
            })
            
        return speakers
            
    def assign_speaker_names(self, speaker_map):
        """Replace generic speaker labels with actual names."""
        if not self.speakers:
            return ""
            
        self.update_status("Assigning speaker names...")
        updated_transcript = []
        
        for segment in self.speakers:
            speaker_id = segment["speaker"]
            speaker_name = speaker_map.get(speaker_id, speaker_id)
            # Format with speaker name in bold
            updated_transcript.append(f"{speaker_name}: {segment['text']}")
            
        result = "\n\n".join(updated_transcript)
        self.transcript = result  # Update the transcript with speaker names
        self.update_status("Speaker names assigned.")
        return result

# LLM Processing Class
class LLMProcessor:
    def __init__(self, client, config_manager, update_callback=None):
        self.client = client
        self.config_manager = config_manager
        self.update_callback = update_callback
        self.chat_history = []
        
    def update_status(self, message):
        if self.update_callback:
            wx.CallAfter(self.update_callback, message)
            
    def generate_response(self, prompt, temperature=None):
        """Generate a response from the LLM."""
        if temperature is None:
            temperature = self.config_manager.get_temperature()
            
        model = self.config_manager.get_model()
        messages = self.prepare_messages(prompt)
        
        try:
            self.update_status("Generating response...")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            
            response_text = response.choices[0].message.content
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": response_text})
            
            self.update_status("Response generated.")
            return response_text
            
        except Exception as e:
            self.update_status(f"Error generating response: {str(e)}")
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
        self.update_status("Chat history cleared.")
        
    def summarize_transcript(self, transcript, template_name=None):
        """Summarize a transcript, optionally using a template."""
        if not transcript:
            return "No transcript to summarize."
            
        self.update_status("Generating summary...")
        
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
                
            self.update_status(f"Summary generated and saved to {summary_filename}.")
            return summary
            
        except Exception as e:
            self.update_status(f"Error generating summary: {str(e)}")
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
        self.audio_processor = AudioProcessor(client, self.update_status)
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
        self.update_status("Application ready.")
        
        # Display info about supported audio formats
        wx.CallLater(1000, self.show_format_info)
        
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
        
        self.identify_speakers_btn = wx.Button(panel, label="Identify Speakers in Transcript")
        self.identify_speakers_btn.SetFont(button_font)
        self.identify_speakers_btn.SetBackgroundColour(wx.Colour(220, 230, 255))  # Light blue background
        self.identify_speakers_btn.Bind(wx.EVT_BUTTON, self.on_identify_speakers)
        speaker_sizer.Add(self.identify_speakers_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        # Add a help text
        help_text = wx.StaticText(panel, label="Click above to detect and label different speakers in your transcript")
        speaker_sizer.Add(help_text, 0, wx.CENTER | wx.ALL, 5)
        
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
        
    def update_status(self, message):
        """Update the status bar with a message."""
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
            self.update_status(f"Selected audio file: {os.path.basename(path)} ({file_size_mb:.1f}MB)")
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
        
        # Update status message
        self.update_status(f"Transcribing in {lang_selection}...")
        
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
            wx.CallAfter(self.update_status, f"Transcription complete: {len(self.audio_processor.transcript)} characters")
            
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
            wx.CallAfter(self.update_status, "Ready")
            
    def show_speaker_id_hint(self):
        """Show a hint dialog about using speaker identification."""
        dlg = wx.MessageDialog(
            self,
            "Transcription is complete!\n\n"
            "To identify different speakers in this transcript, click the 'Identify Speakers' button.\n\n"
            "This will analyze the transcript and tag different speakers.",
            "Speaker Identification",
            wx.OK | wx.ICON_INFORMATION
        )
        dlg.ShowModal()
        dlg.Destroy()
        
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
        """Thread function for speaker identification."""
        try:
            # Show processing message
            wx.CallAfter(self.update_status, "Processing speaker identification...")
            if progress_dialog:
                wx.CallAfter(progress_dialog.Update, 20, "Preparing transcript for analysis...")
                
            # Small delay to ensure the UI updates
            time.sleep(0.5)
            
            if progress_dialog:
                wx.CallAfter(progress_dialog.Update, 40, "Analyzing speech patterns and conversation flow...")
            
            # Force the dialog to update before the potentially lengthy API call
            wx.Yield()
            
            speakers = self.audio_processor.identify_speakers(self.audio_processor.transcript)
            
            if progress_dialog:
                wx.CallAfter(progress_dialog.Update, 80, "Creating speaker mapping interface...")
            
            # Clear existing mapping UI
            wx.CallAfter(self.speaker_mapping_sizer.Clear, True)
            
            # Create UI for speaker mapping
            if speakers:
                wx.CallAfter(self.create_speaker_mapping_ui, speakers)
                speaker_count = len(set(s["speaker"] for s in speakers))
                wx.CallAfter(self.update_status, f"Speaker identification complete. Found {speaker_count} speakers.")
                if progress_dialog:
                    wx.CallAfter(progress_dialog.Update, 100, f"Found {speaker_count} speakers in the transcript!")
            else:
                wx.CallAfter(self.update_status, "No speakers identified.")
                if progress_dialog:
                    wx.CallAfter(progress_dialog.Update, 100, "No speakers were identified in the transcript.")
        except Exception as e:
            if progress_dialog:
                wx.CallAfter(progress_dialog.Destroy)
            wx.CallAfter(wx.MessageBox, f"Speaker identification error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            wx.CallAfter(self.identify_speakers_btn.Enable)
            wx.CallAfter(self.update_button_states)
            
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
            self.update_status("Speaker names applied to transcript.")
        
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
            self.template_content.SetValue(templates[template_name])
        else:
            self.template_content.Clear()
            
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
            self.selected_template.SetItems(templates)
            self.selected_template.SetSelection(templates.index(template_name))
            
            # Clear content
            self.template_content.Clear()
            
        dlg.Destroy()
        
    def on_save_template(self, event):
        """Save the current template."""
        templates = list(self.config_manager.get_templates().keys())
        if not templates:
            wx.MessageBox("No templates available. Create a new template first.", "No Templates", wx.OK | wx.ICON_INFORMATION)
            return
            
        template_name = self.selected_template.GetString(self.selected_template.GetSelection())
        template_content = self.template_content.GetValue()
        
        # Update template
        self.config_manager.add_template(template_name, template_content)
        
        wx.MessageBox(f"Template '{template_name}' saved.", "Template Saved", wx.OK | wx.ICON_INFORMATION)
        
    def on_delete_template(self, event):
        """Delete the selected template."""
        templates = list(self.config_manager.get_templates().keys())
        if not templates:
            return
            
        template_name = self.selected_template.GetString(self.selected_template.GetSelection())
        
        # Confirm deletion
        dlg = wx.MessageDialog(self, f"Are you sure you want to delete the template '{template_name}'?",
                              "Confirm Deletion", wx.YES_NO | wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_YES:
            # Delete template
            self.config_manager.remove_template(template_name)
            
            # Update template choice
            templates = list(self.config_manager.get_templates().keys())
            self.selected_template.SetItems(templates if templates else ["No templates"])
            self.selected_template.SetSelection(0)
            
            # Clear content
            self.template_content.Clear()
            
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
            
            self.update_status("FFmpeg required for M4A support - please install it")
            
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