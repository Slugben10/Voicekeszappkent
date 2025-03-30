# Audio Processing App

This application provides tools for transcribing audio files, identifying speakers, and summarizing content using OpenAI's API services.

## Features

- Transcribe audio files using OpenAI's Whisper API
- Identify different speakers in conversations using two methods:
  - Text analysis (using linguistic patterns)
  - **New!** Voice-based diarization (using pyannote.audio)
- Summarize transcripts with customizable templates
- Support for multiple languages
- Customizable speaker naming

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Install FFmpeg for audio processing:
   - **Windows**: Download from [FFmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (Fedora/CentOS)

## Setting Up Voice-Based Speaker Diarization

For improved speaker detection using audio voice patterns:

1. Install the required packages:
```bash
pip install pyannote.audio torch
```

2. Create a Hugging Face account at [huggingface.co](https://huggingface.co)

3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

4. Accept the user agreement for the diarization model:
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Click "Access Repository" and accept the terms

5. In the app, go to Settings and enter your Hugging Face token

6. Enable "Use voice diarization for speaker identification"

## Usage

1. Launch the application with `python main.py`
2. Enter your OpenAI API key in the Settings tab
3. Upload an audio file and click "Transcribe"
4. After transcription completes, click "Identify Speakers"
5. Customize speaker names if needed
6. Click "Apply Speaker Names" to update the transcript
7. Use "Generate Summary" to create a summary of the conversation

## Supported Audio Formats

- WAV, MP3, M4A, FLAC, MP4, OGG, and other common formats
- Files must be under 25MB (OpenAI API limitation)

## Requirements

- Python 3.7+
- OpenAI API key
- FFmpeg (for handling various audio formats)
- Hugging Face account & token (for voice-based speaker identification)

## Troubleshooting

### M4A File Issues
Some M4A files may have compatibility issues. Try:
- Installing FFmpeg
- Converting to WAV or MP3 format manually

### Voice Diarization Not Working
- Verify pyannote.audio is installed
- Check your Hugging Face token is entered correctly
- Ensure you've accepted the model terms on Hugging Face
- Use a GPU for faster processing (not required but recommended) 