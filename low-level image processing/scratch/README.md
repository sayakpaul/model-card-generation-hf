---
license: mit
tags:
  - stable-diffusion
  - stable-diffusion-diffusers
  - image-to-image
  - art
widget:
  - src: >-
      https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/derain%20the%20image_1.png
    prompt: derain the image
datasets:
- instruction-tuning-sd/low-level-image-proc
---

# Instruction-tuned Stable Diffusion for Low-level Image Processing (Scratch) 

This pipeline is an 'instruction-tuned' version of

## Pipeline description

Motivation behind this pipeline partly comes from [FLAN](https://huggingface.co/papers/2109.01652) and partly
comes from [InstructPix2Pix](https://huggingface.co/papers/2211.09800). The main idea is to first create an
instruction prompted dataset (as described in [our blog](https://hf.co/blog/instruction-tuning-sd)) and then conduct InstructPix2Pix style
training. The end objective is to make Stable Diffusion better at following specific instructions
that entail image transformation related operations.

<p align="center">
<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instruction-tuning-sd.png width=600/>
</p>

Follow [this post](https://hf.co/blog/instruction-tuning-sd) to know more. 

## Training procedure and results

Training was conducted on [instruction-tuning-sd/low-level-image-proc](https://huggingface.co/datasets/instruction-tuning-sd/low-level-image-proc) dataset. Refer to
[this repository](https://github.com/sayakpaul/instruction-tuned-sd) to know more. 

Here are some results dervied from the pipeline:

(TODO)

## Intended uses & limitations

You can use the pipeline for performing Low-level Image Processing.lower() with an input image and an input prompt.

### How to use

Here is how to use this model:

```python
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

model_id = "instruction-tuning-sd/scratch-low-level-img-proc"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=True
).to("cuda")

image_path = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/derain%20the%20image_1.png"
image = load_image(image_path)

image = pipeline("derain the image", image=image).images[0]
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
