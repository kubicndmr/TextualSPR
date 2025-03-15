<h1 align="center">A Textual Surgical Phase Recognition Approach and Enhancement Through Synthetic Surgical Dialogue Generation</h1>

![Figure 1](https://github.com/kubicndmr/TextualSPR/raw/master/Meta/Figure_1.png)

## Getting Started

To install dependencies, we recommend creating a virtual environment as following:
```
    - python3 -m venv surgenv
    - source surgenv/bin/activate
    - pip install -r requirements.txt
```

Create a ```.env``` file in the same directory and copy your personal ```HF_Token```.

Similarly setup your ```Weights & Biases``` connection if you want to use. Remove corresponding lines in ```spr.py``` otherwise.

## Data  

For the **Surgical Phase Recognition (SPR)** task, you need two datasets:  

- **PoCaP (Port Catheter Placement dataset)** – Contains real surgical conversations.  
- **SynPoCaP (Synthetic Port Catheter Placement dataset)** – Contains synthetic surgical conversations generated for pretraining.  

### Setting Up the Data  

1. Copy the **PoCaP** and **SynPoCaP** datasets to a local folder.
2. If you want to generate additional synthetic data, use the `sdg.py` script:  

## Synthetic Data Generation

To start the generation of the synthetic data, use ```sdg.py``` file. This script allows you to specify the number of samples to generate and the target directory for saving the output.

Run the script with the following command:
```python sdg.py -t GeneratedData/ -n 10 -p 5```

Set ```target_path```, ```num_target``` (number of desired synthetic data), and ```prefix_index``` (index of starting synthetic data) according to your needs. 

## Surgical Phase Recognition  

The `spr.py` script is used to train a model for **Surgical Phase Recognition (SPR)**. It supports training with both real and synthetic datasets, allowing fine-tuning and evaluation with configurable hyperparameters.  

### Usage  

Run the script with the following example command:  

```
python spr.py --learning_rate 0.001 --weight_decay 1e-5 --model_dropout 0.3 \
              --model_dim 128 --real_data_path /path/to/real_data \
              --syn_data_path /path/to/synthetic_data --real_dataset_size 36 \
              --syn_dataset_size 50 --eval_ratio 0.25 --n_splits 4 --batch_size 512
```

## Notes
If this study is useful for you, please cite as:

```TBD```





