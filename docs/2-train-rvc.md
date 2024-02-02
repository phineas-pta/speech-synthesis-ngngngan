## Part 2: e.g. of RVC training + inference

original: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

improved fork: https://github.com/Mangio621/Mangio-RVC-Fork

i used: https://github.com/IAHispano/Applio

**6h30min audio (1 speaker) at 48 kHz** + RMVPE pitch extraction = 16.1 GiB disk space

pretrained base models (Discriminator & Generator) v2

hop length only relevant to “crepe” algorithm

train feature index (independent of actual model training)

how to resume training from previous ckpt: increase number of epochs

max batch size depends on VRAM: e.g. 6 if 6 GiB VRAM

during training, monitor 2 losses: G total & D total, save ckpt every 5 epochs so can stop early before overfitting

train 300 epochs but keep ckpt at 160th epoch, see [loss curve](tensorboard/plot_tensorboard_RVC.ipynb)

save model to share:
- save voice for inference only
- save D & G so other can resume training

inference parameters:
- volume envelope = 0
- protect voiceless consonants = 0.5 to disable
- search feature ratio: 3 intervals to test: <0.4, 0.4-0.7, >0.7
- autotune more relevant if singing
- median filtering only relevant to “harvest” algorithm
