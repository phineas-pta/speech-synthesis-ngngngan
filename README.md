# speech-synthesis NgNgNgan

python script to download & process data to train a speech-synthesis model of Vietnamese M.C. Nguyễn Ngọc Ngạn

tải và xử lí audio để train neural network nhái giọng bác Ngạn

vì lí do bản quyền nên ở đây chỉ có code ko có data, ai muốn thì đọc hướng dẫn dưới đây để chạy code kéo audio về tự train

![license](https://www.gnu.org/graphics/agplv3-with-text-162x68.png)

RVC checkpoints: https://huggingface.co/doof-ferb/rvc-ngngngan

Matcha-TTS checkpoints: https://huggingface.co/doof-ferb/matcha_ngngngan

Demo: Matcha-TTS 🤗 https://huggingface.co/spaces/doof-ferb/MatchaTTS_ngngngan

## requirements

need NVIDIA GPU

install `ffmpeg`

`git clone` this repo

prepare a fresh python env (`venv` or `conda`)<br />
`pip install torch torchaudio --find-links=https://download.pytorch.org/whl/torch_stable.html`<br />
optional: `pip install jupyter-lab tensorboard` for visualization<br />
e.g. `tensorboard --logdir <path to folder containing events.out.tfevents.*>` ⇒ `localhost:6006`

or directly run `pip install -r requirements.txt` but it may not be up-to-date

## workflow

[Part 1](docs/1-prepare-data-rvc.md): prepare data for RVC

[Part 2](docs/2-train-rvc.md): e.g. of RVC training + inference

[Part 3](docs/3-prepare-data-vits2.md): prepare data for text-to-speech

[Part 4.1](docs/4-1-train-vits2.md): e.g. VITS 2 training (GIVE UP because training too long)

[Part 4.2](docs/4-2-train-matchatts.md): e.g. Matcha-TTS training

## miscellaneous

```
git update-index --skip-worktree data/vits2_ngngngan_nosdp.json
git update-index --skip-worktree tensorboard/export_tensorboard_RVC.py
git update-index --skip-worktree tensorboard/export_tensorboard_MatchaTTS.py
```
