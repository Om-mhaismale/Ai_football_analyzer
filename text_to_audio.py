# text_to_audio.py
# text_to_audio.py
import pyttsx3
import tempfile
import os

def generate_audio(commentary):
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        temp_path = f.name
    engine.save_to_file(commentary, temp_path)
    engine.runAndWait()
    
    # Read and return the audio as bytes
    with open(temp_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    # Optional: delete temp file
    os.remove(temp_path)

    return audio_bytes
