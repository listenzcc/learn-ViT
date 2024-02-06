"""
File: main.py
Author: Chuncheng Zhang
Date: 2024-01-30
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-01-30 ------------------------
# Requirements and constants
import torch
from transformers import AutoImageProcessor, ViTModel

import random
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from rich import print, inspect

print(f'Cuda availability: {torch.cuda.is_available()}')

# %% ---- 2024-01-30 ------------------------
# Function and class
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", local_files_only=True)
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", local_files_only=True)


# %% ---- 2024-01-30 ------------------------
# Play ground
image = Image.open(random.choice(list(Path('assets/_fullSize').iterdir())))
image = image.convert(mode='RGB')
image


# %% ---- 2024-01-30 ------------------------
# Pending
# Generate 1 x 3 x 224 x 224 pixels image
# If the *image* is an array of n x images,
# the dimension is n x 3 x 224 x 224
inputs = image_processor(image, return_tensors="pt")
inputs['output_hidden_states'] = True
inputs['output_attentions'] = True
inputs['pixel_values'].shape

# %%
with torch.no_grad():
    outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(list(last_hidden_states.shape))
plt.imshow(last_hidden_states[0])
plt.show()

# %% ---- 2024-01-30 ------------------------
# Pending

outputs.hidden_states[0].shape
# %%

# %%
