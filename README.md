# Thesa: A Therapy Chatbot
Thesa is an experimental project of a therapy chatbot trained on mental health data and fine-tuned with the Zephyr GPTQ model that uses quantization to decrease high computatinal and storage costs.

## Table of Contents
1. [Dataset(s)](#datasets)
2. [Training the model](#training-the-model)
3. [How to use](#how-to-use)

## Dataset(s)
At the moment, Thesa is trained with two datasets:
- [CounselChat](https://huggingface.co/datasets/loaiabdalslam/counselchat) - extracted from HuggingFace
- [Mental Health Conversational Data](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data) - extracted from Kaggle

Future plans include expanding training dataset.

## Training the model
### Hardware
Due to lack of computational resources, Thesa has been trained on Google Colab Pro, using an A100 GPU.

### Model architecture
- Fine-tuned from [TheBloke/zephyr-7B-alpha-GPTQ](https://huggingface.co/TheBloke/zephyr-7B-alpha-GPTQ)
- Model configuraton in `model_config.txt`
- Training arguments in `training_args.txt`


## How to use
The resulting fine-tuned model is hosted in HuggingFace at [johnhandleyd/thesa](https://huggingface.co/johnhandleyd/thesa).

You can test if by opening the Jupyter Notebook and refering to the "Inference with some samples" section.

## Results
If you want to see some quick samples, check out `results.md`.