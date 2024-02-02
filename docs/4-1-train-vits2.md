## Part 4.1: e.g. VITS 2 training

**GIVE UP because training too long, switch to Matcha-TTS much faster**

text-to-speech using https://github.com/p0p4k/vits2_pytorch

edit its `vits2_pytorch/requirements.txt` to include `torch` with cuda, `underthesea`, `num2words`<br />
(require MSVC to build Monotonic Alignment Search)

replace content of file `vits2_pytorch/text/cleaners.py` with my file `speech-synthesis-ngngngan/scripts/_cleaner.py`

edit file `vits2_pytorch/utils.py`:
- change `logging.DEBUG` to `logging.WARNING`
- in function `remove_old_checkpoints` remove `print` because it disturbs `tqdm` progress bar

because NCCL not support on windows, must edit file `vits2_pytorch/train.py`:
- replace `mp.spawn(…)` with `run(0, n_gpus, hps)`
- remove `dist.init_process_group(…)`
- remove any `DDP()`
- replace `███.module.███` with  `███.███`

train: `python train.py --model vits2_ngngngan --config ../speech-synthesis-ngngngan/data/vits2_ngngngan_nosdp.json`

config file `vits2_ngngngan_nosdp.json` inspired from
- https://github.com/p0p4k/vits2_pytorch/blob/main/configs/vits2_ljs_nosdp.json
- use newer implementation of duration discriminator (`dur_disc_2`) in p0p4k/vits2_pytorch#59

continue training from the only vietnamese checkpoints i found, shared in https://github.com/p0p4k/vits2_pytorch/pull/10#issuecomment-1724752596 (trained 111k steps but didn’t have duration discriminator) ⇒ 1 h/epoch instead of 2 h/epoch if train from scratch

logs and checkpoints saved in folder `vits2_pytorch/logs/vits2_ngngngan`
