## Part 1: prepare data for RVC

### 0. preview data

get list of audios and info (speakers count):
```
pip install yt-dlp
python scripts/00-get-info.py
```
small test with 2 audio files: `yt-dlp "KgxWziSHQP8" "01WRW7IV1uQ" -x --audio-format "wav" -o "%(id)s.%(ext)s" -P "data/raw"`

### 1. download audio

if good to go then download all youtube audios: `python scripts/01-download-audio.py`<br />
for simplicity, **download only audio with 1 speaker** so after remove silence can go directly to train RVC

audios are saved as `.wav` files in folder `data/01-raw`

### 2. remove silence and non-speech

using SileroVAD: `python scripts/02-remove-silence.py`

audios are saved as `.wav` files in folder `data/02-vad`

### 3. diarize (skip if only 1 speaker)

**TODO**: finalize this step because 1-speaker audio data isn’t big enough for TTS

see https://github.com/pyannote/pyannote-audio to accept ToS of https://huggingface.co/pyannote

then create huggingface token then
```
huggingface-cli login --token=███
pip install pyannote.audio
python scripts/03-diarization.py
```
audios are cut per speaker in folder `data/03-diarized` → listen carefully to each segment (speakers count sometimes not reliable) then remove segments not Nguyễn Ngọc Ngạn

do diarization before voice isolation because `demucs` on large audio raise out-of-memory errors

### 4. isolate voice from noise/music (skip because useless / i see no effect)

using `demucs`:
```
pip install demucs
python scripts/04-isolate-voice.py
```
audios saved in folder `data/04-voices`

### 5. merge segments with same speaker (skip if only 1 speaker)

**TODO**: finalize this step because 1-speaker audio data isn’t big enough for TTS

*DID NOT FINISH as skipped*

merge segments together: `python scripts/05-merge-segments.py`

audios saved in folder `data/05-merged`

### next

go to part 2 to train RVC or continue to part 3 to train TTS
