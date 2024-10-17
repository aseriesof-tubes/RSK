# Learning from Convolution-Based Unlearnable Datasets

This repository contains the implementation of the paper **Learning from Convolution-Based Unlearnable Datasets**. 

It was accepted to the NeurIPS Workshop, **The 3rd New Frontiers in Adversarial Machine Learning 2024** (AdvML Frontiers @ NeurIPS 2024). 

Dohyun Kim, Pedro Sandoval-Segura

## File Descriptions

-   `main.py`: The main script for training the model.
-   `augments.py`: Contains functions for data augmentation, including RSK and FF.
-   `resnet.py`: Implements ResNet-18
-   `util.py`: Utility functions
-   `requirements.txt`: Lists the Python dependencies for the project.

## Requirements

The project requires the following Python packages:

```
torch
torchvision
matplotlib
numpy
tqdm
argparse
Pillow
torch_dct
```

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

## Usage

To train the model with default settings on CIFAR-10 dataset:

```
python main.py
```
This will train a ResNet-18 on the CUDA dataset with the RSK + FF transform applied. 

### Command-line Arguments

You can customize the training process using various command-line arguments. Here are some key options:

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

This command will train the model for 100 epochs on CIFAR-100 dataset using RSK augmentation without CUDA dataset poisoning.

