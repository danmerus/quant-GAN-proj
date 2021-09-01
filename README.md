# Quant GAN

Model implementation of Quant-GAN for Deep Generative Models course at HSE. 
Full demonstration available via Jupyter notebook: `quant.ipynb`.

Based on: https://arxiv.org/pdf/1907.06673.pdf


## Prerequisites

- torch==1.9.0+cu102
- numpy==1.19.5
- pandas==0.24.2
- scipy==1.6.1
- scikit_learn==0.24.2

## Train

Run with `train.py FILENAME.csv`. To tune the hyperparameters, one can specify via argument options, which can been seen via the -h flag.
Resulting generator and discriminator models saved to `QuantGenerator.pt` and `QuantDiscriminator.pt`

## Dataset

Dataset provided `sp_data.csv` is S&P 500 stock prices from may 2009 till dec 2019.


