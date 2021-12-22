

# X-Ray Baggage Scanner 자동 검출 솔루션
<p align="center"><img src="https://user-images.githubusercontent.com/55044675/146106205-337bca43-eefc-4822-9d6b-c467214ca20d.png"></p>

<p align="center">
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/></a>
  <a href="https://www.pytorchlightning.ai/">
    <img src="https://img.shields.io/badge/PyTorch Lightning-792EE5?style=flat-square&logo=PyTorch Lightning&logoColor=white"/></a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white"/></a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=FastAPI&logoColor=white"/></a>
</p>

| [강지우](https://github.com/jiwoo0212) | [곽지윤](https://github.com/kwakjeeyoon) | [서지유](https://github.com/JiyouSeo) | [송나은](https://github.com/sne12345) | [오재환](https://github.com/jaehwan-AI) | [이준혁](https://github.com/kmouleejunhyuk) | [전경재](https://github.com/ppskj178) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| <img src="https://user-images.githubusercontent.com/68782183/146319428-ea9b3554-53d3-46e3-aa41-a0a07660fbab.png" width=800 height=100> | <img src="https://user-images.githubusercontent.com/68782183/146319494-b789dff2-a2c4-49a1-a3f0-29eb5e3f3cf7.png" width=800 height=100> | <img src="https://avatars.githubusercontent.com/u/61641072?v=4" width=800 height=100> | <img src="https://user-images.githubusercontent.com/68782183/138638320-19b24d42-6014-4042-b443-cbeb50251cfd.jpg" width=800 height=100> | <img src="https://user-images.githubusercontent.com/68782183/138295480-ca0169cd-5c40-44ae-b222-d74d9cc4bc82.jpg" width=800 height=100> | <img src="https://user-images.githubusercontent.com/68782183/146321291-46ede634-7371-4d3e-9ccd-0932ad3fee7b.png" width=800 height=100> | <img src="https://user-images.githubusercontent.com/20790778/138396418-b669cbed-40b0-45eb-9f60-7167cae739b7.png" width=800 height=100> | |


## Project overview
현재 사람이 직접 위해물품을 탐지하는 시스템에서는 human inspection error로 인해 80~90%의 정확도를 보입니다. 그리하여 AI 기술을 활용한 X-Ray Baggage Scanner 자동 검출 솔루션을 통해 더 높은 정확도를 보장하고 다양한 품목에 대하여 검출 가능하도록 하였습니다. 

## Project roadmap
![그림2](https://user-images.githubusercontent.com/49234207/147053477-26a6edd2-5ba4-45cf-bab2-f5290ca34286.png)




## Dataset 
- [AI Hub 공항 유해물품 xray 데이터](https://aihub.or.kr/aidata/33)
<img src="https://aihub.or.kr/sites/default/files/inline-images/%EB%8C%80%ED%91%9C%EB%8F%84%EB%A9%B4_1.png">


## Contents
```
Root
├── app
├── data
├── detection
├── Dockerfile
├── Makefile
├── multilabel
├── README.md
├── requirements.txt
├── setup.py
└── utils
    └── timer.py
```
- app/ : web resource
- data/ : train data path (mount dataset path)
- detection/ : detection model path
- multilabel/ : multilabel classification model path
- Makefile : Web Server Entrypoint
- Dockerfile : Dockerfile 


## Result
- 기존 20개 위해품목 감지
  - 38개 전체 위해품목 감지 가능
- 기존 정확도 90% 이상 
  - (Multilabel)EMR 기준 0.99 이상 
  - (Object detection) mAP 기준 0.9 이상
- 기존 inference time 0.7s/image
  - under 0.2s/image (with GPU)
  - under 5s/image (without GPU)
    - GPU 추가 없이 소프트웨어 설치만으로 이용 가능한 수준


## Test setting
- CPU : Intel(R) Xeon(R) Gold 5120 CPU(8 core)
- GPU : Tesla V100-PCIE-32gb
