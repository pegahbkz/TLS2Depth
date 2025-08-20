# TLS2Depth
This repository contains the code for improving outdoor depth estimation using millimetre-accurate Terrestrial Laser Scan data from the [Oxford Spires dataset and Depth Anything V2.

## Setup

### Prerequisites
- Python 3.x
- Required dependencies (install with `pip install -r requirements.txt`)

### Data and Checkpoint Setup
You must populate the checkpoint and data directories from this link: https://drive.google.com/drive/folders/1qxLPyPEk6EMqybnhJF9d9CGcG8tkWSLG?usp=drive_link

### Training
To start finetuning, run:
```bash
python finetuning/train.py
