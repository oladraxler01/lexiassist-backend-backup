from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import speech_recognition as sr
from pydub import AudioSegment
import os
import uvicorn
import uuid
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="LexiAssist Audio Service",
    description="Speech-to-Text using SpeechRecognition + pydub (supports ALL formats)",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create recognizer
recognizer = sr.Recognizer()

# Create temp directory
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Pydantic models
class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: str = "default"
    speed: float = 1.0

class SpeechToTextResponse(BaseModel):
    text: str
    confidence: float
    language: str
    original_format: str

class TextToSpeechResponse(BaseModel):
    audio_file_url: str
    message: str

# Health check
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "audio",
        "port": 5004,
        "version": "2.0.0",
        "engine": "SpeechRecognition + pydub",
        "supported_formats": ["mp3", "wav", "m4a", "ogg", "mp4", "webm", "flac", "aac"]
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "SpeechRecognition + pydub",
        "features": {
            "speech_to_text": "available (all formats)",
            "text_to_speech": "placeholder"
        }
    }

def convert_to_wav(input_path: str, output_path: str):
    """
    Convert ANY audio format to WAV using pydub
    """
    try:
        # Load audio file (pydub auto-detects format)
        audio = AudioSegment.from_file(input_path)

        # Export as WAV (required for speech_recognition)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

@app.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(
    audio: UploadFile = File(..., description="Audio file (MP3, WAV, M4A, OGG, etc.)"),
    language: str = Form("en-US", description="Language code (en-US, es-ES, fr-FR, etc.)")
):
    """
    Convert uploaded audio file to text.
    Supports: MP3, WAV, M4A, OGG, MP4, WEBM, FLAC, AAC
    """
    input_path = None
    wav_path = None

    try:
        # Get file extension
        file_ext = os.path.splitext(audio.filename)[1].lower()

        # Save uploaded file
        temp_id = str(uuid.uuid4())[:8]
        input_path = os.path.join(TEMP_DIR, f"input_{temp_id}{file_ext}")
        wav_path = os.path.join(TEMP_DIR, f"converted_{temp_id}.wav")

        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        print(f"\n🎤 Processing audio: {audio.filename}")
        print(f"   Format: {file_ext}")
        print(f"   Size: {len(content)} bytes")
        print(f"   Language: {language}")

        # Convert to WAV (if not already WAV)
        if file_ext == '.wav':
            wav_path = input_path
            print("   Already WAV format")
        else:
            print(f"   Converting {file_ext} to WAV...")
            success = convert_to_wav(input_path, wav_path)
            if not success:
                raise HTTPException(status_code=400, detail=f"Could not convert {file_ext} to WAV")
            print("   ✅ Conversion successful")

        # Process with speech_recognition
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        # Use Google Speech Recognition
        text = recognizer.recognize_google(audio_data, language=language)

        print(f"   ✅ Transcription: {text[:100]}...")

        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        if wav_path != input_path and os.path.exists(wav_path):
            os.remove(wav_path)

        return SpeechToTextResponse(
            text=text,
            confidence=0.95,
            language=language,
            original_format=file_ext
        )

    except sr.UnknownValueError:
        # Cleanup on error
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        raise HTTPException(status_code=400, detail="Could not understand audio. Try speaking clearly or check audio quality.")

    except sr.RequestError as e:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        raise HTTPException(status_code=500, detail=f"Google Speech API error: {str(e)}")

    except Exception as e:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/text-to-speech", response_model=TextToSpeechResponse)
async def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech. Placeholder for production TTS.
    """
    return TextToSpeechResponse(
        audio_file_url="placeholder.mp3",
        message="TTS not implemented. Use Google Cloud TTS, gTTS, or ElevenLabs for production."
    )

@app.get("/languages")
async def list_languages():
    """
    List supported languages for speech recognition.
    """
    languages = {
        "en-US": "English (US)",
        "en-GB": "English (UK)",
        "es-ES": "Spanish",
        "fr-FR": "French",
        "de-DE": "German",
        "it-IT": "Italian",
        "pt-BR": "Portuguese (Brazil)",
        "ja-JP": "Japanese",
        "zh-CN": "Chinese (Simplified)",
        "ko-KR": "Korean",
        "ar-SA": "Arabic",
        "hi-IN": "Hindi",
        "ru-RU": "Russian",
        "auto": "Auto-detect"
    }
    return {"supported_languages": languages}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5004, reload=True)
