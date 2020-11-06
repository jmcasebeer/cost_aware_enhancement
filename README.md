# Communication Aware Microphone Selection For Neural Speech Enhancement with Ad-hoc Microphone Arrays

This repo contains the code to reproduce our results and experiments. Our dataset is based off of the [microsoft scalable noisy speech dataset](https://github.com/microsoft/MS-SNSD).

## Dataset Setup

First, you will need to clone the microsoft scalable noisy speech dataset with:

```bash
git clone https://github.com/microsoft/MS-SNSD.git
```
Then, you will need to edit the path to the microsoft dataset in the desired config within the data_gen_config.py file. Once this is done, you can run the data generation script with:


```bash
python data_gen.py --cfg <name of config>
```

This can take a while to run. Once it has finished generating then you need to edit the paths in your config in train_config.py to point to this new dataset. 

## Model Training
The model can then be trained with:

```bash
python train.py --cfg <name of config> --gpu <which gpu to run on>
```

To edit the model paramters you can change the config in train_config.py. Make sure that the config points to the dataset you generated earlier.

The website associated with the repo can be found [here](https://jmcasebeer.github.io/projects/camse/).

