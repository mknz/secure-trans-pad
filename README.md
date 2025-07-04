# secure-trans-pad
**This tool is still experimental!**

Secure local transcription and translation tool using [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) with output to CryptPad or to the console

## Requirements
```
sudo apt-get install pulseaudio alsa-utils portaudio19-dev -y
```
Install [uv](https://docs.astral.sh/uv/) and git clone this repo.

## Usage
### Translate to English
```
uv run main.py --url CRYPTPAD_URL --mode translate-whisper
```
### Transcribe
```
uv run main.py --url CRYPTPAD_URL --mode transcribe
```
### Explicitly specify the source language
```
uv run main.py --url CRYPTPAD_URL --lang de
```
### Output the text only to the console
```
uv run main.py
```
### Choose model type (default is `small`)
```
uv run main.py --model medium
```
### Translate the text using [llm](https://github.com/simonw/llm)
```
uv run main.py --mode translate-llm \
               --model-translate gemma3:1b \
               --translation-prompt prompt_de.txt
```
### Keep audio files
```
uv run main.py --keep
```
### Show delay
```
uv run main.py --show-delay
```

Consider making a donation to [CryptPad](https://cryptpad.fr/)!
