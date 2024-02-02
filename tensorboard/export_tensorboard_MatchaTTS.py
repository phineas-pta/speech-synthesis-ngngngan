#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""export tensorboard data to csv for visualization purpose"""

from glob import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

data = {
	"train loss by step":  {"step":  [], "value": []},
	  "val loss by step":  {"step":  [], "value": []},
	"train loss by epoch": {"epoch": [], "value": []},
	  "val loss by epoch": {"epoch": [], "value": []},
}

def update_data(select: str, scale: str, EventAcc: EventAccumulator) -> None:
	for i in EventAcc.Scalars(f"loss/{select}_{scale}"):
		data[f"{select} loss by {scale}"][scale  ].append(i.step)
		data[f"{select} loss by {scale}"]["value"].append(i.value)

# https://tbparse.readthedocs.io/en/latest/pages/raw.html
for file in glob("<path to Matcha-TTS>/logs/matcha_ngngngan/tensorboard/version_0/events.out.tfevents.*"):
	event_acc = EventAccumulator(file)
	event_acc.Reload()
	# print(event_acc.Tags())  # to see available monitored values
	update_data("train", "step",  event_acc)
	update_data("val",   "step",  event_acc)
	update_data("train", "epoch", event_acc)
	update_data("val",   "epoch", event_acc)

for k, v in data:
	pd.DataFrame(v).to_csv(f"data_tensorboard_MatchaTTS ({k}).csv", index=False)
