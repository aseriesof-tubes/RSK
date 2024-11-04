# Learning from Convolution-Based Unlearnable Datasets

This repository contains the implementation of the paper **Learning from Convolution-Based Unlearnable Datasets**. 

It was accepted to the NeurIPS Workshop, **The 3rd New Frontiers in Adversarial Machine Learning 2024** (AdvML Frontiers @ NeurIPS 2024). 

Dohyun Kim, Pedro Sandoval-Segura

## Use RSK and SSK + FF
If you want to use our SSK + FF or RSK + FF transforms, use the following code: 
```python
RSK_FF = transforms.Compose([ToTensor(), 
         transforms.Lambda(lambda x: sharpen_image(x, center_mean=sharp_center, random=True)),
         transforms.Lambda(lambda x: get_dct_image(x, ff_percentage))])
SSK_FF = transforms.Compose([ToTensor(), 
         transforms.Lambda(lambda x: sharpen_image(x, center_mean=sharp_center, random=False)), 
         transforms.Lambda(lambda x: get_dct_image(x, ff_percentage))])
```
and copy the `sharpen_image` and `get_dct_image` functions from `augments.py`.

## Usage

You can customize the training process using various command-line arguments. Here are the options:

-   `--epochs`: Number of training epochs (default: 60)
-   `--dataset`: Choose between 'cifar10' and 'cifar100' (default: 'cifar10')
-   `--normalize`: Normalize the dataset (after transforms) (default: True)
-   `--ud`: Choose between 'cuda' (for CUDA dataset) and 'clean' (default: 'cuda')
-   `--blur_parameter`: Blur parameter for CUDA dataset (default: 0.3)
-   `--sharp-center`: Changes the center value for the sharpening kernel (default: 2.5)
-   `--ff_percentage`: Sets the percentage of lowest frequencies kept (default: 30) 
-   `--transform`: Augmentation technique to use (choices: 'ssk', 'ssk_ff', 'rsk', 'rsk_ff', 'ff', 'clean', default: 'rsk_ff')

Example:

```
python main.py --epochs 100 --dataset cifar100 --transform rsk --ud clean
```

This command will train a ResNet-18 for 100 epochs on the CIFAR-100 dataset using RSK augmentation without CUDA dataset poisoning.

To train the model with default settings on CIFAR-10 dataset:

```
python main.py
```
This will train a ResNet-18 on the CUDA dataset with the RSK + FF transform applied. 

## File Descriptions

-   `main.py`: The main script for training the model.
-   `augments.py`: Contains functions for data augmentation, including RSK and FF.
-   `resnet.py`: Implements ResNet-18
-   `util.py`: Utility functions
-   `requirements.txt`: Lists the Python dependencies for the project.

## Installation

1. Create a conda environment:

   ```
   conda create -n rsk python=3.10
   conda activate rsk
   ```

2. Clone this repository:

   ```
   git clone https://github.com/aseriesof-tubes/rsk.git
   cd rsk
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
