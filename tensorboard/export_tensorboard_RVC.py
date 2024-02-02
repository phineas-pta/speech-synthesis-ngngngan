#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""export tensorboard data to csv for visualization purpose"""

from glob import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

data = {
	"step": [],
	"lossG": [],
	"lossD": [],
}

# https://tbparse.readthedocs.io/en/latest/pages/raw.html
for file in glob("<path to Applio-RVC-Fork>/logs/nguyenngocngan/events.out.tfevents.*"):
	event_acc = EventAccumulator(file)
	event_acc.Reload()
	# print(event_acc.Tags())  # to see available monitored values
	data["step"].extend([i.step for i in event_acc.Scalars("learning_rate")])  # get step not learning rate
	data["lossG"].extend([i.value for i in event_acc.Scalars("loss/g/total")])
	data["lossD"].extend([i.value for i in event_acc.Scalars("loss/d/total")])

pd.DataFrame(data).to_csv("data_tensorboard_RVC.csv", index=False)
