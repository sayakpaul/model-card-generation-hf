# Model card generation for Hugging Face Hub 🤗

Model card is an important component to consider when releasing a model. You
can know more about model cards [here](https://huggingface.co/course/chapter4/4?fw=pt).

Hugging Face Hub houses thousands of different models. Models that are a part of organizations such as 
[Google](https://hf.co/google/), [Facebook](https://hf.co/facebook/), [NVIDIA](https://hf.co/nvidia/),
[Hugging Face](https://hf.co/huggingface/), etc. all have model cards. Apart from that, other models contributed
by the community also come with model cards (thanks to the
[`Trainer` class](https://huggingface.co/docs/transformers/main_classes/trainer) that makes this easy). 

This repository shows how to generate model cards for a specific family of model in bulk. 

## Considerations

* Prepare the generic text following [these guidelines](https://huggingface.co/course/chapter4/4?fw=pt). The
text will remain fixed for all the different variants of a particular model family
(BERT is a model family and BERT-Base is a variant).
* Determine the variables that will change in the different model cards. Some
examples: (pre-)training dataset, model architecture, parameters, performance metrics, etc. 

This repository demonstrates these considerations with the MAXIM [1] model family. Here the
generic text can be found in `template` of the `generate_model_card.py` script. The
variables are found in the `Config` class:

* `dataset_metadata`
* `task` 
* `dataset` 
* `input_url` 
* `ckpt` 
* `psnr` 
* `ssim`

## Model card generation

Modify the `generate_model_card.py` script as per your needs and then run it `python generate_model_card.py`. 

If you run the `generate_model_card.py` as is here, you should see 11 directories
(with the `maxim-s` prefix) after successful execution. 

## References

[1] [MAXIM: Multi-Axis MLP for Image Processing](https://arxiv.org/abs/2201.02973)

## Acknowledgement

Thanks to Willi Gierke from Google who initially gave me an earlier version of this script.
I used that to generate TF-Hub documentation for
[30 different variants of the ConvNeXt model](https://tfhub.dev/sayakpaul/collections/convnext/1).