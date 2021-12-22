# Multi-label classification

## Content
```
├── API
│   ├── model.py
│   ├── OOD.py
│   ├── preprocess.py
│   └── __pycache__
├── baseline
│   ├── config
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── multilabel_utils
│   ├── optim_sche.py
│   ├── __pycache__
│   ├── resources.py
│   ├── retrain.py
│   ├── save
│   ├── train.py
│   ├── transform.py
│   ├── twostage_train.py
│   └── weights
├── multi_tree.txt
└── requirements.txt
```


## Model
- Backbone Model
  
  ResNet101
<img src="https://user-images.githubusercontent.com/68782183/147064277-73dcc696-a07b-4bc9-b47d-b1a833c1946a.png" height="200" width="300">



- Multi Head


<img src="https://user-images.githubusercontent.com/68782183/147063530-8a44018f-c371-44c7-9b5f-07a74cafce3c.png" height="200" width="400">


<br>

## Best Result

- Single Head
Valid EMR 0.72 
[weight, config download](https://drive.google.com/drive/folders/1uxmlhF2mXmXu6fvWVMNOa2cUnbl03j3A?usp=sharing)

- Multi Head
Valid EMR 0.77 
[weight, config download](https://drive.google.com/drive/folders/1kgg-KwT5aHRM-6gfg8mBafL-qUn7DbGH?usp=sharing)

- Multi Head - Full data train
Valid EMR 0.99 
[weight, config download](https://drive.google.com/drive/folders/1LhFXnXA9X9VEE6SFroAIAVcViE0pvRdT?usp=sharing)

<br>

## Simple Start

Train
```
python train.py --config_train "config.yaml"
```
