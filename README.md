# AIM-AHEAD-Challenge-2023
HI-PHIVE submission to the AIM-AHEAD Health Equity Challenge in 2023

### Brief Method Description 
A ResNet50 is trained to classify individual tiles into their stage. These tiles are randomly selected from a WSI for a patient and given the stage label corresponding to the patient. During testing, 20 tiles are sampled from the patient and ran through the trained network. The stage predictions from the 10 randomly selected tiles are then mean-pooled to come to a final stage prediction per biopsy, then pooled again to come to a final prediction per patient. `training_tile.py` works from a file which had the coords for each tile in their own row, this corresponds to our second submission. `training.py` corresponds to our first submission and contains code working from my splits made from the Nightingale-provided csvs. 
