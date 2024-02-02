# -*- coding: utf-8 -*-

from torchaudio import load as _load_file, info as _read_info, save as _save_file
from torchaudio.functional import resample as _resample


def load_audio(filepath: str) -> dict:
	"""return the format required by pyannote pipeline"""
	waveform, sample_rate = _load_file(filepath)
	metadata = _read_info(filepath)
	return {
		"waveform": waveform,  # waveform.shape = [2 audio channels, audio length in sec × sampling rate]
		"sample_rate": sample_rate,  # 48 kHz for all audio downloaded
		"bits_per_sample": metadata.bits_per_sample,  # default 16 for wav
		"encoding": metadata.encoding,  # default "PCM_S" for wav
	}


def cut_audio_timestamp(audio_file: dict, outfile: str, start: float, end: float) -> None:
	"""cut audio with timestamp from start to end"""

	start_frame = int(start * audio_file["sample_rate"])
	end_frame   = int(  end * audio_file["sample_rate"])
	cut_waveform = audio_file["waveform"][:, start_frame:end_frame]

	_save_file(
		outfile, cut_waveform,
		sample_rate=audio_file["sample_rate"],
		bits_per_sample=audio_file["bits_per_sample"],
		encoding=audio_file["encoding"]
	)


_LJspeech_rate = 22050
def cut_audio_timestamp_vits2(audio_file: dict, outfile: str, start: float, end: float) -> None:
	"""cut audio but downsampling and convert dual channel to mono for training data for VITS 2"""

	start_frame = int(start * audio_file["sample_rate"])
	end_frame   = int(  end * audio_file["sample_rate"])
	cut_waveform = audio_file["waveform"][:, start_frame:end_frame]

	# downloaded audio is dual channel 48 kHz, TTS training data only mono channel 22.05 kHz (see LJ Speech dataset)
	mono_wav = cut_waveform.mean(dim=0, keepdim=True)
	down_wav = _resample(mono_wav, orig_freq=audio_file["sample_rate"], new_freq=_LJspeech_rate)

	_save_file(
		outfile, down_wav,
		sample_rate=_LJspeech_rate,
		bits_per_sample=audio_file["bits_per_sample"],
		encoding=audio_file["encoding"]
	)
