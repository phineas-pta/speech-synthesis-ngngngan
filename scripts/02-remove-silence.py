#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""remove silence and non-speech using Silero VAD"""
# see https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies
# also https://github.com/snakers4/silero-vad/blob/master/utils_vad.py

import os
from tqdm import tqdm
import torch
from torchaudio import save as _save_audio

from _constants import LIST_VID, RAW_DATA_PATH, VAD_DATA_PATH
from _utils import load_audio

TQDM_PBAR_FORM = "{percentage: 5.1f}% |{bar}| {n:.0f}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
SAMPLING_RATE = 16000  # Silero VAD operating value
MODEL, (get_speech_timestamps, _, silero_read_audio, _, _) = torch.hub.load(
	repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
)
MODEL = MODEL.to("cuda")


def vad_filter(infile: str, outfile: str) -> None:
	wav = silero_read_audio(infile, sampling_rate=SAMPLING_RATE).to("cuda")  # SileroVAD operate on mono channel at 16 kHz
	with torch.inference_mode(), tqdm(total=wav.shape[0], bar_format=TQDM_PBAR_FORM) as pbar:
		speech_timestamps: list[dict[str, int]] = get_speech_timestamps(
			wav, MODEL, sampling_rate=SAMPLING_RATE,
			progress_tracking_callback=lambda val: pbar.update(val * 10)  # weird, TODO: raise issue in silero repo
		)
	torch.cuda.empty_cache()

	# convert timestamps to match original audio file (dual channels & higher bit rate)
	audio_file: dict[str, int | str | torch.Tensor] = load_audio(infile)
	ratio = audio_file["sample_rate"] / SAMPLING_RATE
	cut_waveform = torch.cat([
		audio_file["waveform"][:, int(el["start"] * ratio) : int(el["end"] * ratio)]
		for el in speech_timestamps
	], dim=1)
	_save_audio(
		outfile, cut_waveform,
		sample_rate=audio_file["sample_rate"],
		bits_per_sample=audio_file["bits_per_sample"],
		encoding=audio_file["encoding"]
	)


#################################### main #####################################
for id in LIST_VID.keys():
	infile = os.path.join(RAW_DATA_PATH, f"{id}.wav")
	outfile = os.path.join(VAD_DATA_PATH, f"{id}.wav")
	if not os.path.exists(infile):
		print(f"{id} not found")
	else:
		print(f"{id} to be VAD filtered")
		vad_filter(infile, outfile)
