# This script demonstrates how to use the Google GenAI Live API to interact with the Gemini model using audio.
# It reads a local audio file, sends it to the model, and saves the model's audio response.
#
# This script demonstrates a direct connection: Client (this Python script) -> Google API Server
# There is NO intermediate backend server. This script acts as the "client" and connects directly 
# to Google's Live API servers using WebSockets (handled by the genai.Client SDK).
# This is the "Client-to-server" approach where "server" = Google's API servers.

import os
import asyncio  # Used for asynchronous execution (the API is async)
import io       # Used for in-memory byte buffers
from pathlib import Path
import wave     # Used to write the output audio file (standard WAV format)
from dotenv import load_dotenv

# The official Google GenAI SDK
from google import genai
from google.genai import types

# Audio processing libraries:
# soundfile: Used to write raw audio data into the buffer in the specific PCM format required by the API.
import soundfile as sf
# librosa: Used to load and resample the input audio file to the required 16kHz sample rate.
import librosa


load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# New native audio model:
model = "gemini-2.5-flash-native-audio-preview-09-2025"

config = {
  "response_modalities": ["AUDIO"],
  "system_instruction": "You are a helpful assistant and answer in a friendly tone.",
}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:

        buffer = io.BytesIO()
        # Load and resample the input audio file to the required 16kHz sample rate.
        y, sr = librosa.load("sample.wav", sr=16000)
        # Write the resampled audio to the buffer in the required PCM format.
        sf.write(buffer, y, sr, format='RAW', subtype='PCM_16')
        # Move the cursor to the beginning of the buffer.
        buffer.seek(0)
        # Read the buffer into a byte array.
        audio_bytes = buffer.read()

        # If already in correct format, you can use this:
        # audio_bytes = Path("sample.pcm").read_bytes()

        # Send the audio to the model.
        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )

        # Open the file using 'with', which automatically closes it when done
        with wave.open("audio.wav", "wb") as wf:
            # Set the number of channels to 1 (mono).
            wf.setnchannels(1)
            # Set the sample width to 2 bytes (16-bit).
            wf.setsampwidth(2)
            # Set the frame rate to 24kHz.
            wf.setframerate(24000)

            async for response in session.receive():
                if response.data is not None:
                    wf.writeframes(response.data)

if __name__ == "__main__":
    asyncio.run(main())