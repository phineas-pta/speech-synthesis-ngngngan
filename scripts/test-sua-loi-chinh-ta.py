#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""sửa lỗi chính tả với PhoGPT"""

# pip install -q transformers accelerate bitsandbytes einops
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from _constants import AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP

TRANSCRIPTION_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all.txt")
RAW_DATA = pd.read_csv(TRANSCRIPTION_FILE, sep=FIELD_SEP, names=["audio", "text"])

MODEL_ID = "vinai/PhoGPT-4B-Chat"  # "vinai/PhoGPT-4B-Chat", "vinai/PhoGPT-7B5-Instruct" (7B need login token)
QUANTIZATION = "float16"  # "float16", "8bit", "4bit"

match QUANTIZATION:
	case "float16":
		MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
	case "8bit":
		MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, load_in_8bit=True, device_map="auto", trust_remote_code=True)
	case "4bit":
		BNB_CONFIG = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
		MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=BNB_CONFIG, device_map="auto", trust_remote_code=True)
	case _:
		raise ValueError("oh no!")

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL_CONFIG = GenerationConfig(
	max_new_tokens=1024, do_sample=True,
	top_p=.95, top_k=40, temperature=.1, repetition_penalty=1.05,
	eos_token_id=TOKENIZER.eos_token_id, pad_token_id=TOKENIZER.pad_token_id,
)

def sua_chinh_ta(txt: str) -> str:
	text = "### Câu hỏi: Sửa lỗi chính tả:\n" + txt + "\n### Trả lời:"
	input_ids = TOKENIZER(text, return_tensors="pt").to(MODEL.device)
	out_ids = MODEL.generate(**input_ids, generation_config=MODEL_CONFIG)
	answer = TOKENIZER.batch_decode(out_ids, skip_special_tokens=True)[0]
	return answer.strip().split("### Trả lời:")[-1]

RAW_DATA["new_text"] = [sua_chinh_ta(txt) for txt in tqdm(RAW_DATA["text"])]  # RAW_DATA["text"].map(sua_chinh_ta) but no progress bar
RAW_DATA.to_csv("_all_test.txt", columns=["audio",  "new_text"], sep="|", index=False, header=False, encoding="utf-8")
