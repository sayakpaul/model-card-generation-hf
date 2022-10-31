"""Generates model cards for MAXIM TF models.

Thanks to Willi Gierke.
"""

import os
from string import Template

import attr

from mappings import DATASETS

template = Template(
    """---
license: apache-2.0
library_name: keras
language: en
tags:
- vision
- maxim
- image-to-image
datasets:
- $DATASET_METADATA
---

# MAXIM pre-trained on $DATASET for $TASK 

MAXIM model pre-trained for $TASK. It was introduced in the paper [MAXIM: Multi-Axis MLP for Image Processing](https://arxiv.org/abs/2201.02973) by Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, Yinxiao Li and first released in [this repository](https://github.com/google-research/maxim). 

Disclaimer: The team releasing MAXIM did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

MAXIM introduces a shared MLP-based backbone for different image processing tasks such as image deblurring, deraining, denoising, dehazing, low-light image enhancement, and retouching. The following figure depicts the main components of MAXIM:

![](https://github.com/google-research/maxim/raw/main/maxim/images/overview.png)

## Training procedure and results

The authors didn't release the training code. For more details on how the model was trained, refer to the [original paper](https://arxiv.org/abs/2201.02973). 

As per the [table](https://github.com/google-research/maxim#results-and-pre-trained-models), the model achieves a PSNR of $PSNR and an SSIM of $SSIM. 

## Intended uses & limitations

You can use the raw model for $TASK tasks. 

The model is [officially released in JAX](https://github.com/google-research/maxim). It was ported to TensorFlow in [this repository](https://github.com/sayakpaul/maxim-tf). 

### How to use

Here is how to use this model:

```python
from huggingface_hub import from_pretrained_keras
from PIL import Image

import tensorflow as tf
import numpy as np
import requests

url = "$INPUT_URL"
image = Image.open(requests.get(url, stream=True).raw)
image = np.array(image)
image = tf.convert_to_tensor(image)
image = tf.image.resize(image, (256, 256))

model = from_pretrained_keras("$CKPT")
predictions = model.predict(tf.expand_dims(image, 0))
```

For a more elaborate prediction pipeline, refer to [this Colab Notebook](https://colab.research.google.com/github/sayakpaul/maxim-tf/blob/main/notebooks/inference-dynamic-resize.ipynb). 

### Citation

```bibtex
@article{tu2022maxim,
  title={MAXIM: Multi-Axis MLP for Image Processing},
  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},
  journal={CVPR},
  year={2022},
}
```
"""
)


@attr.s
class Config:
    dataset_metadata = attr.ib(type=str)
    task = attr.ib(type=str)
    dataset = attr.ib(type=str)
    input_url = attr.ib(type=str)
    ckpt = attr.ib(type=str)
    psnr = attr.ib(type=float)
    ssim = attr.ib(type=float)

    def get_folder_name(self):
        return self.ckpt.split("/")[-1]


for c in [
    Config(
        "sidd",
        "image denoising",
        DATASETS["sidd"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Denoising/input/0011_23.png",
        "google/maxim-s3-denoising-sidd",
        39.96,
        0.960,
    ),
    Config(
        "realblur_r",
        "image deblurring",
        DATASETS["realblur_r"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Deblurring/input/1fromGOPR0950.png",
        "google/maxim-s3-deblurring-realblur-r",
        39.45,
        0.962,
    ),
    Config(
        "realblur_j",
        "image deblurring",
        DATASETS["realblur_j"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Deblurring/input/1fromGOPR0950.png",
        "google/maxim-s3-deblurring-realblur-j",
        32.84,
        0.935,
    ),
    Config(
        "gopro",
        "image deblurring",
        DATASETS["gopro"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Deblurring/input/1fromGOPR0950.png",
        "google/maxim-s3-deblurring-gopro",
        32.86,
        0.961,
    ),
    Config(
        "rain13k",
        "image deraining",
        DATASETS["rain13k"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Deraining/input/55.png",
        "google/maxim-s2-deraining-rain13k",
        33.24,
        0.933,
    ),
    Config(
        "raindrop",
        "image deraining",
        DATASETS["raindrop"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Deraining/input/55.png",
        "google/maxim-s2-deraining-raindrop",
        31.87,
        0.935,
    ),
    Config(
        "sots-indoor",
        "image dehazing",
        DATASETS["sots-indoor"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Dehazing/input/1440_10.png",
        "google/maxim-s2-dehazing-sots-indoor",
        38.11,
        0.991,
    ),
    Config(
        "sots-outdoor",
        "image dehazing",
        DATASETS["sots-outdoor"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Dehazing/input/0048_0.9_0.2.png",
        "google/maxim-s2-dehazing-sots-outdoor",
        34.19,
        0.985,
    ),
    Config(
        "lol",
        "image enhancement",
        DATASETS["lol"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Enhancement/input/748.png",
        "google/maxim-s2-enhancement-lol",
        23.43,
        0.863,
    ),
    Config(
        "fivek",
        "image retouching",
        DATASETS["fivek"],
        "https://github.com/sayakpaul/maxim-tf/raw/main/images/Enhancement/input/748.png",
        "google/maxim-s2-enhancement-fivek",
        26.15,
        0.945,
    ),
]:
    model_folder = c.get_folder_name()
    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

    save_path = os.path.join(model_folder, "README.md")

    with open(save_path, "w") as f:
        f.write(
            template.substitute(
                DATASET=c.dataset,
                DATASET_METADATA=c.dataset_metadata,
                TASK=c.task,
                INPUT_URL=c.input_url,
                CKPT=c.ckpt,
                PSNR=c.psnr,
                SSIM=c.ssim,
            )
        )
