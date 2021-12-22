

# X-Ray Baggage Scanner 자동 검출 솔루션
<p align="center"><img src="https://user-images.githubusercontent.com/55044675/146106205-337bca43-eefc-4822-9d6b-c467214ca20d.png"></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/></a>
  <img src="https://img.shields.io/badge/PyTorch Lightning-792EE5?style=flat-square&logo=PyTorch Lightning&logoColor=white"/></a>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white"/></a>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=FastAPI&logoColor=white"/></a>
</p>

| [강지우](https://github.com/jiwoo0212) | [곽지윤](https://github.com/kwakjeeyoon) | [서지유](https://github.com/JiyouSeo) | [송나은](https://github.com/sne12345) | [오재환](https://github.com/jaehwan-AI) | [이준혁](https://github.com/kmouleejunhyuk) | [전경재](https://github.com/ppskj178) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| <img src="https://user-images.githubusercontent.com/68782183/146319428-ea9b3554-53d3-46e3-aa41-a0a07660fbab.png" width=800 height=100> | <img src="https://user-images.githubusercontent.com/68782183/146319494-b789dff2-a2c4-49a1-a3f0-29eb5e3f3cf7.png" width=800 height=100> | <img src="https://avatars.githubusercontent.com/u/61641072?v=4" width=800 height=100> | <img src="https://user-images.githubusercontent.com/68782183/138638320-19b24d42-6014-4042-b443-cbeb50251cfd.jpg" width=800 height=100> | <img src="https://user-images.githubusercontent.com/68782183/138295480-ca0169cd-5c40-44ae-b222-d74d9cc4bc82.jpg" width=800 height=100> | <img src="https://user-images.githubusercontent.com/68782183/146321291-46ede634-7371-4d3e-9ccd-0932ad3fee7b.png" width=800 height=100> | <img src="https://user-images.githubusercontent.com/20790778/138396418-b669cbed-40b0-45eb-9f60-7167cae739b7.png" width=800 height=100> | |


## Project overview
- 공항에 입출국할 때 스캔된 결과물을 사람이 직접 판독을 하는데, 사람이 직접 판독하기 때문에 장시간 업무로 인한 스트레스와 피로감으로 실수가 발생합니다. 
- 이 때문에 사람의 정확도는 80에서 90%로 그렇게 높지 않고, 실수도 자주 발생합니다.
- 그래서 저희는 자동 검출 솔루션을 통해 휴먼 inspection 에러를 줄일 수 있다고 판단을 해서 이 프로젝트를 진행했습니다.

## Project roadmap
![그림2](https://user-images.githubusercontent.com/49234207/147051585-6bbad261-c8b2-4948-baec-5d0ef7ad7aa5.png)



## Dataset 
- [AI Hub 공항 유해물품 xray 데이터](https://aihub.or.kr/aidata/33)
<img src="https://aihub.or.kr/sites/default/files/inline-images/%EB%8C%80%ED%91%9C%EB%8F%84%EB%A9%B4_1.png">


## Content
```
Root
├── app
├── data -> /opt/ml/data/data
├── detection
├── Makefile
├── multilabel
├── README.md
├── requirements.txt
├── setup.py
└── utils
    ├── __pycache__
    └── timer.py

14 directories, 19 files
```
- app/
- data/
- detection/
- multilabel/
- utils/
- Makefile




