# Multi-label classification
## Pytorch
[<img alt="alt_text" width="300px" src="https://user-images.githubusercontent.com/49234207/147069604-fd4d5a52-e52f-450d-8ff6-9fbb6872a072.png" />](https://pytorch.org)


## Content
```
root/multilabel
├── API
│   ├── model.py
│   ├── OOD.py
│   └── preprocess.py 
├── baseline
│   ├── config
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── multilabel_utils
│   ├── optim_sche.py
│   ├── resources.py
│   ├── retrain.py
│   ├── save
│   ├── train.py
│   ├── inference.ipynb
│   ├── transform.py
│   └── weights
└── requirements.txt
```


## Model

- Backbone Model: ResNet101  
![그림4](https://user-images.githubusercontent.com/49234207/147072747-caf33f94-b21d-4cf6-b42e-1e06b5f336dd.png)



- Multi Head  
![그림3](https://user-images.githubusercontent.com/49234207/147073247-c5b9443a-0d57-4334-bc10-1df9bc586c09.png)


<br>

## Inference Result & weights
```
- Valid EMR : 0.99 
- Inference time : under 0.2s (with GPU, without grad-cam)
```
[weight, config download](https://drive.google.com/drive/folders/1LhFXnXA9X9VEE6SFroAIAVcViE0pvRdT?usp=sharing)


![samples](https://user-images.githubusercontent.com/49234207/147074833-c6cbd799-1ec3-4fba-8b94-293e284a7877.png)


<br>

## Simple Start

Train
```
python train.py --config_train "config.yaml"
```
Inference  
```
multilabel/baseline/inference.ipynb
```

## References
[1] https://pytorch.org  
[2] Deep Residual Learning for Image Recognition(2015, He et al)  
[3] Tao, Siyan & Guo, Yao & Zhu, Chuang & Chen, Huang & Zhang, Yue & Yang, Jie & Liu, Jun. (2019). Highly Efficient Follicular Segmentation in Thyroid Cytopathological Whole Slide Image.
