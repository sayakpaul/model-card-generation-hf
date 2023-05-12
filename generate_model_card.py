"""Generates model cards for MAXIM TF models.

Thanks to Willi Gierke.
"""

import os
from string import Template

import attr


template = Template(
    """---
license: mit
tags:
  - stable-diffusion
  - stable-diffusion-diffusers
  - image-to-image
  - art
widget:
  - src: >-
      $INFERENCE_IMG
    prompt: $PROMPT
datasets:
- $DATASET
---

# Instruction-tuned Stable Diffusion for $TASK ($VARIANT) 

$TRAINING_TEXT

## Pipeline description

Motivation behind this pipeline partly comes from [FLAN](https://huggingface.co/papers/2109.01652) and partly
comes from [InstructPix2Pix](https://huggingface.co/papers/2211.09800). The main idea is to first create an
instruction prompted dataset (as described in [our blog](https://hf.co/blog/instruction-tuning-sd)) and then conduct InstructPix2Pix style
training. The end objective is to make Stable Diffusion better at following specific instructions
that entail image transformation related operations.

<p align="center">
<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instruction-tuning-sd.png" width=600/>
</p>

Follow [this post](https://hf.co/blog/instruction-tuning-sd) to know more. 

## Training procedure and results

Training was conducted on [$DATASET](https://huggingface.co/datasets/$DATASET) dataset. Refer to
[this repository](https://github.com/sayakpaul/instruction-tuned-sd) to know more. 

Here are some results dervied from the pipeline:

(TODO)

## Intended uses & limitations

You can use the pipeline for performing $TASK.lower() with an input image and an input prompt.

### How to use

Here is how to use this model:

```python
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

model_id = "$MODEL_ID"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=True
).to("cuda")

image_path = "$INFERENCE_IMG"
image = load_image(image_path)

image = pipeline("$PROMPT", image=image).images[0]
image.save("image.png")
```

For notes on limitations, misuse, malicious use, out-of-scope use, please refer to the model card
[here](https://huggingface.co/runwayml/stable-diffusion-v1-5).

## Citation

**FLAN**

```bibtex
@inproceedings{
    wei2022finetuned,
    title={Finetuned Language Models are Zero-Shot Learners},
    author={Jason Wei and Maarten Bosma and Vincent Zhao and Kelvin Guu and Adams Wei Yu and Brian Lester and Nan Du and Andrew M. Dai and Quoc V Le},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=gEZrGCozdqR}
}
```

**InstructPix2Pix**

```bibtex
@InProceedings{
    brooks2022instructpix2pix,
    author     = {Brooks, Tim and Holynski, Aleksander and Efros, Alexei A.},
    title      = {InstructPix2Pix: Learning to Follow Image Editing Instructions},
    booktitle  = {CVPR},
    year       = {2023},
}
```

**Instruction-tuning for Stable Diffusion blog**

```bibtex
@article{
  Paul2023instruction-tuning-sd,
  author = {Paul, Sayak},
  title = {Instruction-tuning Stable Diffusion with InstructPix2Pix},
  journal = {Hugging Face Blog},
  year = {2023},
  note = {https://huggingface.co/blog/instruction-tuning-sd},
}
```
"""
)


@attr.s
class Config:
    inference_img = attr.ib(type=str)
    dataset = attr.ib(type=str)
    training_text = attr.ib(type=str)
    task = attr.ib(type=str)
    variant = attr.ib(type=float)
    model_id = attr.ib(type=float)
    prompt = attr.ib(type=float)

    def get_folder_name(self):
        return os.path.join(self.task.lower(), self.variant.lower())

SCRATCH_TXT = "This pipeline is an 'instruction-tuned' version of [Stable Diffusion (v1.5)](https://huggingface.co/runwayml/stable-diffusion-v1-5). It was trained using the [InstructPix2Pix methodology](https://huggingface.co/papers/2211.09800)."
FINETUNING_TXT = "This pipeline is an 'instruction-tuned' version of [Stable Diffusion (v1.5)](https://huggingface.co/runwayml/stable-diffusion-v1-5). It was fine-tuned from the existing [InstructPix2Pix checkpoints](https://huggingface.co/timbrooks/instruct-pix2pix).",


for c in [
    Config(
        "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png",
        "instruction-tuning-sd/cartoonization",
        SCRATCH_TXT,
        "Cartoonization",
        "Scratch",
        "instruction-tuning-sd/scratch-cartoonizer",
        "Cartoonize the following image",

    ),
    Config(
        "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png",
        "instruction-tuning-sd/cartoonization",
        FINETUNING_TXT,
        "Cartoonization",
        "Fine-tuned",
        "instruction-tuning-sd/cartoonizer",
        "Cartoonize the following image",
    ),
    Config(
        "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/derain%20the%20image_1.png",
        "instruction-tuning-sd/low-level-image-proc",
        SCRATCH_TXT,
        "Low-level Image Processing",
        "Scratch",
        "instruction-tuning-sd/scratch-low-level-img-proc",
        "derain the image",
    ),
    Config(
        "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/derain%20the%20image_1.png",
        "instruction-tuning-sd/low-level-image-proc",
        FINETUNING_TXT,
        "Low-level Image Processing",
        "Fine-tuned",
        "instruction-tuning-sd/scratch-low-level-img-proc",
        "derain the image",
    ),
]:
    model_folder = c.get_folder_name()
    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

    save_path = os.path.join(model_folder, "README.md")

    with open(save_path, "w") as f:
        f.write(
            template.substitute(
                INFERENCE_IMG=c.inference_img,
                DATASET=c.dataset,
                TRAINING_TEXT=c.training_text,
                TASK=c.task,
                VARIANT=c.variant,
                MODEL_ID=c.model_id,
                PROMPT=c.prompt
            )
        )
