# Lecture-Transcription-Summarization

This repository provides scripts for automating video transcription and lecture summarization. It extracts audio from videos, transcribes it using the Faster-Whisper model, and summarizes lecture notes using the Gemini LLM model. The summaries are then converted to HTML for easy study and reference.

## Features

- **Video Transcription**: Extracts audio from video files and generates text transcriptions.
- **Lecture Summarization**: Summarizes lecture transcriptions using the Gemini LLM model.
- **HTML Conversion**: Converts summaries into a styled HTML format for study.
- **Batch Processing**: Supports processing multiple videos in a folder.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Lecture-Transcription-Summarization.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key for the Google Gemini model in `note_from_transcription.py`.

## Usage

1. **Transcribe Videos**:  
   Call the `video_transcribe.py` script to extract audio and transcribe video files:
   ```python
   python video_transcribe.py
   ```

2. **Summarize Transcriptions**:  
   Use `note_from_transcription.py` to summarize a lecture transcription:
   ```python
   python note_from_transcription.py
   ```

3. **Convert Summary to HTML**:  
   Summaries will be automatically converted to HTML for easy review.

## Contributing

Feel free to fork the repository, create issues, or submit pull requests.

## License

This project is licensed under the MIT License.