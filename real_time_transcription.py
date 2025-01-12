import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import wave
from queue import Queue
from threading import Thread
import time

class RealtimeTranscriber:
    def __init__(self, model_size="base"):
        # Initialize Whisper model
        self.model = WhisperModel(model_size, device="cuda", compute_type="int8")
        
        # Audio parameters
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024 * 4  # Larger chunk size for better processing
        self.RECORD_SECONDS = 3  # Process audio in 3-second segments
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Create a queue for audio chunks
        self.audio_queue = Queue()
        
        # Flag to control recording
        self.is_recording = False

    def callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        """Process audio chunks and transcribe"""
        while self.is_recording:
            # Collect audio for RECORD_SECONDS
            audio_data = []
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                if not self.is_recording:
                    break
                if not self.audio_queue.empty():
                    audio_data.append(self.audio_queue.get())

            if audio_data:
                # Combine all chunks
                audio_segment = np.concatenate(audio_data)
                
                # Transcribe
                segments, _ = self.model.transcribe(
                    audio_segment, 
                    beam_size=5,
                    language="en",
                    vad_filter=True
                )
                
                # Print transcription
                for segment in segments:
                    print(f"Transcription: {segment.text}")

    def start_transcription(self):
        """Start real-time transcription"""
        self.is_recording = True
        
        # Start audio stream
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.callback
        )

        # Start processing thread
        self.process_thread = Thread(target=self.process_audio)
        self.process_thread.start()
        
        print("Started recording and transcribing... Press Ctrl+C to stop")

    def stop_transcription(self):
        """Stop real-time transcription"""
        self.is_recording = False
        
        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()
        
        # Wait for processing thread to finish
        self.process_thread.join()
        
        # Clean up PyAudio
        self.audio.terminate()
        print("\nStopped recording and transcribing")

# Example usage
if __name__ == "__main__":
    transcriber = RealtimeTranscriber(model_size="base")
    try:
        transcriber.start_transcription()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        transcriber.stop_transcription()