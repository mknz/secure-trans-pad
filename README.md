# secure-trans-pad
***This tool is still experimental!***

Secure local transcription and translation tool using Faster Whisper with output to CryptPad

## Requirements
Install [uv](https://docs.astral.sh/uv/) and git clone this repo.

## Usage
### Translate to English
```
uv run main.py --url CRYPTPAD_URL --task translate
```
### Transcribe
```
uv run main.py --url CRYPTPAD_URL --task transcribe
```
### Explicitly specify the source language
```
uv run main.py --url CRYPTPAD_URL --lang de
```
### Output the text only to the console
```
uv run main.py
```

Consider making a donation to [CryptPad](https://cryptpad.fr/)!
