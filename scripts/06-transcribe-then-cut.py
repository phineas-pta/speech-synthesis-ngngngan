#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""transcribe audio, then cut audio into segments"""

import os
import torch
import whisper  # `pip install openai-whisper` not other whisper packages on PyPI

from _constants import LIST_VID, VAD_DATA_PATH, SUBS_DATA_PATH, AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP
from _utils import load_audio, cut_audio_timestamp_vits2

# i don’t have a gpu supported by flash-attention 2 (which still not work on windows) so cannot use huggingface transformers
# so i quantize: see https://github.com/MiscellaneousStuff/openai-whisper-cpu
MODEL = whisper.load_model("large", device="cpu")  # not enough VRAM to load directly
# directly quantize may throw out-of-memory error if not enough RAM
MODEL.encoder = torch.quantization.quantize_dynamic(MODEL.encoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8).to("cuda")
MODEL.decoder = torch.quantization.quantize_dynamic(MODEL.decoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8).to("cuda")


HALLUCINATIONS_TEXT = "hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn"

@torch.inference_mode()
def transcribe(infile: str) -> list[dict]:
	"""transcribe, and if get non-sense transcribe again"""
	res = MODEL.transcribe(infile, verbose=False, language="vi")["segments"]  # verbose=False to show progress bar
	torch.cuda.empty_cache()
	if all(chunk["text"].strip().lower() == HALLUCINATIONS_TEXT for chunk in res):  # 2 files: m4N42aD6Twg ELbEiuSHSoE
		print(" "*8, "totally garbage transcription, going to transcribe again")
		res = MODEL.transcribe(infile, verbose=False, language="vi", condition_on_previous_text=False)["segments"]
		torch.cuda.empty_cache()
	return res


def cut_audio_and_save_text(infile: str, res_trans: list[dict], file_id: str, outdir: str, text_file_buffer) -> None:
	audio_file = load_audio(infile)
	prev_txt = ""  # to check repeated text
	for chunk in res_trans:
		txt = chunk["text"].strip()
		outfile = f"{file_id}_{chunk['id']:04d}.wav"  # can go up to 1k chunks and more
		cut_audio_timestamp_vits2(audio_file, os.path.join(outdir, outfile), chunk["start"], chunk["end"])

		# check whether text can be added to audio-text file-list
		# (still save audio to estimate how much audio is lost later-on)
		tmp = txt.lower()
		if tmp == HALLUCINATIONS_TEXT or tmp == prev_txt:
			print(" "*8, "skip", outfile, "⇐ hallucinated:", txt)
		elif " " not in txt:  # only 1 word
			# if the audio is too small and wrongly annotated the length of the text can exceed the length of the mel spectrogram which
			# results in the breakdown of the Monotonic Alignment Search (MAS) that is used to learn the alignments between text and audio
			print(" "*8, "skip", outfile, "⇐ too short:", txt)
		else:  # add to audio-text file-list
			text_file_buffer.write(outfile + FIELD_SEP + txt + "\n")
			prev_txt = tmp
	# do not use flush() it may cause bad texts


#################################### main #####################################
# initialize the audio-text file-list
TRANSCRIPTION_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all.txt")
with open(TRANSCRIPTION_FILE, mode="w", encoding="utf-8") as f:
	f.write("")

for id in LIST_VID.keys():
	infile = os.path.join(VAD_DATA_PATH, f"{id}.wav")
	if not os.path.exists(infile):
		print(f"{id} not found")
	else:
		print(f"{id} to be transcribed then cut")
		res_trans = transcribe(infile)  # whisper fail to accept dual channel waveform directly
		# transcribing takes much more time than cut audio, so keep text file open may have bad consequences
		with open(TRANSCRIPTION_FILE, mode="a", encoding="utf-8") as f:
			cut_audio_and_save_text(infile, res_trans, id, SUBS_DATA_PATH, f)
		# with this i can follow and verify spelling in near real-time
		# to check for hallucination unnoticed and spelling (e.g. r-d-gi, tr-ch, …)

# risk of hallucinations: to verify in detail: m4N42aD6Twg ELbEiuSHSoE CDlujtWFuJE gp2mK345es0 KpNxXYPYm0I ZbKrts2AtYQ
# ATTENTION: timestamp mismatch: j8YV2HtDZxA p6Pjce5khIY
# ATTENTION: singing: ZbKrts2AtYQ
