# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faster-whisper",
#     "numpy",
#     "playwright",
#     "pyaudio",
# ]
# ///
import argparse
import datetime
import os
import shutil
import tempfile
import time
import wave

import numpy as np
import pyaudio

from faster_whisper import WhisperModel
from faster_whisper.utils import available_models
from playwright.sync_api import sync_playwright


# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024 * 4  # Larger chunk for better transcription
SILENCE_THRESHOLD_MEAN = 300


# Buffer for audio data
audio_buffer = []


def transcribe_audio(task, lang, keep, model):
    global audio_buffer

    temp_dir = tempfile.gettempdir()

    while True:
        if len(audio_buffer) > 0:
            # Copy and clear the buffer
            current_buffer = audio_buffer.copy()
            audio_buffer = []

            # Save buffer to temp WAV file
            temp_file = os.path.join(
                temp_dir,
                f"segment_{time.time()}.wav",
            )
            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 2 bytes for paInt16
                wf.setframerate(RATE)
                wf.writeframes(b"".join(current_buffer))

            # Transcribe audio
            segments, _ = model.transcribe(
                temp_file,
                language=lang,
                task=task,
                beam_size=5,
            )

            # Get transcription text
            result = []
            for segment in segments:
                # Crude VAD
                if segment.no_speech_prob < 0.5:
                    result.append(segment.text)
            text = ' '.join(result)

            # Clean up
            try:
                if keep:
                    fn = f'{datetime.datetime.now().isoformat()}.wav'
                    shutil.move(temp_file, fn)
                else:
                    os.remove(temp_file)
            except Exception:
                print("failed to move or remove tmp file")

            if text.strip():
                # Return text to be written to webpage
                yield text

        # Wait a bit to not overwhelm the system
        time.sleep(0.1)


def audio_callback(in_data, frame_count, time_info, status):
    # Check if audio contains speech (simple amplitude threshold)
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    if (np.abs(audio_data).mean() > SILENCE_THRESHOLD_MEAN):
        audio_buffer.append(in_data)
    return (in_data, pyaudio.paContinue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--model",
        choices=available_models(),
        default="large",
    )

    args = parser.parse_args()
    url = args.url
    task = args.task
    lang = args.lang
    keep = args.keep

    # Initialize Whisper model
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    # Start audio recording
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback,
    )

    stream.start_stream()
    print("Recording started. Speak into the microphone.")

    if url:
        # Initialize Playwright and browser
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            page = browser.new_page()

            page.goto(url)
            (
                page.locator("#sbox-iframe")
                .content_frame.locator('iframe[title="Editor\\, editor1"]')
                .content_frame.locator("html")
                .click()
            )
            # Create a stream to process transcriptions
            try:
                for text in transcribe_audio(task, lang, keep, model):
                    # Update the webpage with the transcribed text
                    elem = (
                        page.locator("#sbox-iframe")
                        .content_frame.locator('iframe[title="Editor\\, editor1"]')
                        .content_frame.locator("html")
                    )
                    elem.evaluate(
                        f"""
                        let body = document.querySelector("body");
                        let p = document.createElement("p");
                        p.textContent = "{text}";
                        body.appendChild(p);
                    """
                    )
                    print(f"Transcribed: {text}")
            except KeyboardInterrupt:
                print("Recording stopped.")

    else:
        # Create a stream to process transcriptions
        try:
            for text in transcribe_audio(task, lang, keep, model):
                print(f"Transcribed: {text}")
        except KeyboardInterrupt:
            print("Recording stopped.")

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
