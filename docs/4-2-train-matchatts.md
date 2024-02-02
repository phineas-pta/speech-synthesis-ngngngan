## Part 4.2: e.g. Matcha-TTS training

text-to-speech using https://github.com/shivammehta25/Matcha-TTS (model with 18.2 millions parameters)

edit its `Matcha-TTS/requirements.txt` to include `torch` with cuda, `underthesea`, `num2words`, replace `torchvision` with `torchaudio`<br />
remove `piper_phonemize` to bypass restriction python 3.10

copy my files
- from `speech-synthesis-ngngngan/scripts/_cleaner.py` to `Matcha-TTS/matcha/text/cleaners.py`
- from `speech-synthesis-ngngngan/scripts/_symbols.py` to `Matcha-TTS/matcha/text/symbols.py`
- from `speech-synthesis-ngngngan/data/matcha_exp_ngngngan.yaml` to `Matcha-TTS/configs/experiment/matcha_ngngngan.yaml` (change `max_epochs` in this file)
- from `speech-synthesis-ngngngan/data/matcha_data_ngngngan.yaml` to `Matcha-TTS/configs/data/matcha_ngngngan.yaml` (change `num_workers` to a value ≤ cpu threads count)

edit all audio-text file-list: change `../speech-synthesis-ngngngan` to absolute path

edit `Matcha-TTS/matcha/cli.py`: change `english_cleaners2` to `basic_cleaners_ngngngan`

install requirements then `pip install -e .`<br />
(require MSVC to build Monotonic Alignment Search)

run `python -m matcha.utils.generate_data_statistics -i matcha_ngngngan.yaml` (remember edit file path)<br />
get 2 values `mel_mean` &amp; `mel_std` then go back edit file `matcha_ngngngan.yaml` (should be correct already)

train: `python -m matcha.train experiment=matcha_ngngngan`

logs and checkpoints saved in folder `Matcha-TTS/logs/matcha_ngngngan` (configured by me - not default value)

to resume training: add `ckpt_path=logs/matcha_ngngngan/checkpoints/last.ckpt`

export onnx: `python -m matcha.onnx.export matcha.ckpt model.onnx --n-timesteps=10`

infer with torch: `python -m matcha.cli --checkpoint_path=… --output_folder=outputs --steps=10 --text=…`

infer with onnx: original code contains error if use cuda
