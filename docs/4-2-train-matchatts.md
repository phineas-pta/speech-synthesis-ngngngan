## Part 4.2: e.g. Matcha-TTS training

text-to-speech using https://github.com/shivammehta25/Matcha-TTS (model with 18.2 millions parameters)

edit its `Matcha-TTS/requirements.txt` to include `underthesea` + `num2words`, remove `torchvision`, remove `piper_phonemize` to bypass restriction python 3.10

copy my files
- from `speech-synthesis-ngngngan/scripts/_cleaner.py` to `Matcha-TTS/matcha/text/cleaners.py`
- from `speech-synthesis-ngngngan/scripts/_symbols.py` to `Matcha-TTS/matcha/text/symbols.py`
- from `speech-synthesis-ngngngan/data/matcha_exp_ngngngan.yaml` to `Matcha-TTS/configs/experiment/matcha_ngngngan.yaml` (change `max_epochs` in this file)
- from `speech-synthesis-ngngngan/data/matcha_data_ngngngan.yaml` to `Matcha-TTS/configs/data/matcha_ngngngan.yaml` (change `num_workers` to a value ≤ cpu threads count)

edit all audio-text file-list: change `../speech-synthesis-ngngngan` to absolute path

edit `Matcha-TTS/matcha/cli.py`: change `english_cleaners2` to `basic_cleaners_ngngngan`

run `pip install -e . --find-links=https://download.pytorch.org/whl/torch_stable.html`<br />
(require MSVC to build Monotonic Alignment Search)

run `python matcha/utils/generate_data_statistics.py -i matcha_ngngngan.yaml` (remember edit file path)<br />
get 2 values `mel_mean` &amp; `mel_std` then go back edit file `matcha_ngngngan.yaml` (should be correct already)

train: `python matcha/train.py experiment=matcha_ngngngan`

logs and checkpoints saved in folder `Matcha-TTS/logs/matcha_ngngngan` (configured by me - not default value)

to resume training: add `ckpt_path=logs/matcha_ngngngan/checkpoints/checkpoint_epoch███.ckpt` (must rename file to remove `=` character)

export onnx: `python matcha/onnx/export.py matcha.ckpt model.onnx --n-timesteps=10`

infer with torch: `python matcha/cli.py --vocoder=hifigan_univ_v1 --checkpoint_path=… --output_folder=outputs --steps=10 --text=…`

infer with onnx: original code contains error if use cuda

### trim down checkpoint

default `pytorch lightning` setting: `save_weights_only: false` to resume training later-on ⇒ but 3× file size (optimizer states, learning rate scheduler states, etc.)

after finish training, keep bare minimum data in checkpoint:
```python
import os, glob, torch
CKPT_DIR = os.path.join("logs", "matcha_ngngngan", "checkpoints")
REQUIRED_INFO = ("state_dict", "hyper_parameters", "epoch", "pytorch-lightning_version")
for f in glob.glob("*.ckpt", root_dir=CKPT_DIR):
	infile = os.path.join(CKPT_DIR, f)
	outfile = os.path.splitext(infile)[0] + "_slim.pt"
	yolo = torch.load(infile)
	# print(list(yolo.keys()))
	torch.save({k: yolo[k] for k in REQUIRED_INFO}, outfile)
```
