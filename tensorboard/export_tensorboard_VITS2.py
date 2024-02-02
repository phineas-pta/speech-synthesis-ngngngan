#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""export tensorboard data to csv for visualization purpose"""

from glob import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

data = {
	"step": [],
	"loss Generator": [],
	"loss Discriminator": [],
	"loss Duration discrimitor": [],
}

# https://tbparse.readthedocs.io/en/latest/pages/raw.html
for file in glob("<path to vits2_pytorch>/logs/vits2_ngngngan/events.out.tfevents.*"):
	event_acc = EventAccumulator(file)
	event_acc.Reload()
	# print(event_acc.Tags())  # to see available monitored values
	data["step"].extend([i.step for i in event_acc.Scalars("learning_rate")])  # get step not learning rate
	data["loss Generator"].extend([i.value for i in event_acc.Scalars("loss/g/total")])
	data["loss Discriminator"].extend([i.value for i in event_acc.Scalars("loss/d/total")])
	data["loss Duration discrimitor"].extend([i.value for i in event_acc.Scalars("loss/dur_disc/total")])

pd.DataFrame(data).to_csv("data_tensorboard_VITS2.csv", index=False)
