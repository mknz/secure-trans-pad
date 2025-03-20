# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faster-whisper",
#     "numpy",
#     "llm",
#     "llm-ollama",
#     "playwright",
#     "pyaudio",
# ]
# ///
import asyncio
import argparse
import datetime
import os
import shutil
import signal
import tempfile
import time
import wave
from typing import Generator

import numpy as np
import pyaudio

from faster_whisper import WhisperModel
from faster_whisper.utils import available_models
import llm
from playwright.async_api import async_playwright


class TranscriptionService:
    # Audio configuration
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024 * 4  # Larger chunk for better transcription
    SILENCE_THRESHOLD_MEAN = 300

    def __init__(self, args):
        self.args = args
        self.audio_buffer = []
        self.transcript_buffer = []
        self.running = True
        self.stream = None
        self.p_audio = None
        self.model = None
        self.temp_dir = tempfile.gettempdir()

        # Set up signal handling for graceful exit
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def handle_exit(self, sig, frame):
        """Handle exit signals gracefully"""
        print("\nShutting down gracefully...")
        self.running = False

        # Clean up resources
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()

        if self.p_audio:
            self.p_audio.terminate()

        print("Resources cleaned up. Exiting.")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Process incoming audio data"""

        audio_data = np.frombuffer(in_data, dtype=np.int16)
        if np.abs(audio_data).mean() > self.SILENCE_THRESHOLD_MEAN:
            self.audio_buffer.append(in_data)
        return (in_data, pyaudio.paContinue)

    async def transcribe_audio(self) -> Generator[str, None, None]:
        """Process audio buffer and transcribe content"""
        while self.running:
            if len(self.audio_buffer) > 0:
                # Copy and clear the buffer
                current_buffer = self.audio_buffer.copy()
                self.audio_buffer = []

                # Save buffer to temp WAV file
                temp_file = os.path.join(
                    self.temp_dir,
                    f"segment_{time.time()}.wav",
                )
                with wave.open(temp_file, "wb") as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(2)  # 2 bytes for paInt16
                    wf.setframerate(self.RATE)
                    wf.writeframes(b"".join(current_buffer))

                # Determine task based on mode
                task = (
                    "translate"
                    if self.args.mode == "translate-whisper"
                    else "transcribe"
                )

                # Transcribe audio
                segments, _ = self.model.transcribe(
                    temp_file,
                    language=self.args.lang,
                    task=task,
                    beam_size=5,
                )

                # Get transcription text
                result = []
                for segment in segments:
                    # Crude VAD
                    if segment.no_speech_prob < 0.5:
                        result.append(segment.text)

                text = " ".join(result)

                # Clean up
                try:
                    if self.args.keep:
                        fn = f"{datetime.datetime.now().isoformat()}.wav"
                        shutil.move(temp_file, fn)
                    else:
                        os.remove(temp_file)
                except Exception:
                    print("Failed to move or remove tmp file")

                if text.strip():
                    # Add to transcript buffer and yield
                    self.transcript_buffer.append(text)
                    yield text

            # Status update
            print(
                f"Buffers: audio={len(self.audio_buffer)}, transcript={len(self.transcript_buffer)}",
                end="\r",
            )
            await asyncio.sleep(0.1)

    async def translate(self) -> None:
        """Translate transcribed text using specified method"""
        last_index = 0

        if self.args.mode == "translate-llm":
            model = llm.get_async_model(self.args.model_translate)
            with open(self.args.translation_prompt, "r") as f:
                prompt = f.read()

            while self.running:
                if last_index < len(self.transcript_buffer):
                    text = self.transcript_buffer[last_index]
                    prompt_text = f"{prompt}\n---\n{text}"
                    output = await model.prompt(prompt_text).text()
                    print(f"\nTranslated: {output}")
                    last_index += 1
                await asyncio.sleep(0.1)

        elif self.args.mode == "translate-whisper":
            while self.running:
                if last_index < len(self.transcript_buffer):
                    text = self.transcript_buffer[last_index]
                    print(f"\nTranslated: {text}")
                    last_index += 1
                await asyncio.sleep(0.1)

    async def update_webpage(self, page, text: str) -> None:
        """Update the webpage with transcribed text"""
        elem = (
            page.locator("#sbox-iframe")
            .content_frame.locator('iframe[title="Editor\\, editor1"]')
            .content_frame.locator("html")
        )
        # Properly escape the text for safe insertion into JavaScript
        escaped_text = text.replace('"', '\\"').replace("\n", "\\n")
        await elem.evaluate(
            f"""
            let body = document.querySelector("body");
            let p = document.createElement("p");
            p.textContent = "{escaped_text}";
            body.appendChild(p);
            """
        )
        # Trigger save
        await asyncio.sleep(0.1)
        await page.keyboard.press('Enter')
        await page.keyboard.press('Backspace')

    async def start_transcription(self) -> None:
        """Initialize and start the transcription process"""
        # Initialize Whisper model
        self.model = WhisperModel(self.args.model, device="cpu", compute_type="int8")

        # Start audio recording
        self.p_audio = pyaudio.PyAudio()
        self.stream = self.p_audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback,
        )

        self.stream.start_stream()
        print("Recording started. Speak into the microphone. Press Ctrl+C to exit.")

        if self.args.url:
            # Use Playwright for web interaction
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=False)
                page = await browser.new_page()

                await page.goto(self.args.url)
                (
                    await page.locator("#sbox-iframe")
                    .content_frame.locator('iframe[title="Editor\\, editor1"]')
                    .content_frame.locator("html")
                    .click()
                )

                try:
                    async for text in self.transcribe_audio():
                        # Update webpage and print progress
                        await self.update_webpage(page, text)
                        if self.args.mode != "translate-whisper":
                            print(f"\nTranscribed: {text}")
                except Exception as e:
                    print(f"\nError: {e}")
                    self.running = False
                finally:
                    await browser.close()
        else:
            # Console-only mode
            try:
                async for text in self.transcribe_audio():
                    if self.args.mode != "translate-whisper":
                        print(f"\nTranscribed: {text}")
            except Exception as e:
                print(f"\nError: {e}")
                self.running = False


async def main():
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription and translation tool"
    )
    parser.add_argument("--url", default=None,)
    parser.add_argument(
        "--mode",
        choices=["transcribe", "translate-whisper", "translate-llm"],
        default="transcribe",
    )
    parser.add_argument("--lang", default=None)
    parser.add_argument(
        "--keep", action="store_true", help="Keep temporary audio files"
    )
    parser.add_argument(
        "--model",
        choices=available_models(),
        default="large",
        help="Whisper models to use",
    )
    parser.add_argument(
        "--model-translate", default=None, help="LLM model id for translation"
    )
    parser.add_argument(
        "--translation-prompt", default=None, help="Path to translation prompt file"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "translate-llm" and (
        not args.model_translate or not args.translation_prompt
    ):
        parser.error(
            "--model-translate and --translation-prompt are required for 'translate-llm' mode"
        )

    # Create service and run tasks
    service = TranscriptionService(args)

    try:
        await asyncio.gather(
            service.start_transcription(),
            service.translate(),
        )
    except asyncio.CancelledError:
        print("\nTasks cancelled")
    finally:
        # Ensure cleanup happens
        if service.stream:
            service.stream.stop_stream()
            service.stream.close()

        if service.p_audio:
            service.p_audio.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
