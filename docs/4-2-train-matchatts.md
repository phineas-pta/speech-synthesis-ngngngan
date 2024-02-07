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

for inference: use my fork (having CLI & gradio GUI): https://github.com/phineas-pta/MatchaTTS_ngngngan

### trim down checkpoint

default `pytorch lightning` setting: `save_weights_only: false` to resume training later-on ⇒ but 3× file size (optimizer states, learning rate scheduler states, etc.)

after finish training, keep bare minimum data in checkpoint:
```python
from os.path import join, splitext
from glob import glob
from tqdm import tqdm
from torch import load, save

CKPT_DIR = join("logs", "matcha_ngngngan", "checkpoints")
REQUIRED_INFO = ("state_dict", "hyper_parameters", "epoch", "pytorch-lightning_version")

for f in tqdm(glob("*.ckpt", root_dir=CKPT_DIR)):
	infile = join(CKPT_DIR, f)
	outfile = splitext(infile)[0] + "_slim.pt"
	yolo = load(infile)
	# print(list(yolo.keys()))
	save({k: yolo[k] for k in REQUIRED_INFO}, outfile)
```
