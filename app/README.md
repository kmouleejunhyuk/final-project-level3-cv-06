## Content
```
app
├── app_config.py
├── app_config.yaml
├── customrouter.py
├── front_config.py
├── front_config.yaml
├── frontend.py
├── __main__.py
├── main.py
├── routes
│   ├── detection_pred.py
│   ├── detection.py
│   ├── __init__.py
│   ├── multilabel_pred.py
│   └── multilabel.py
└── static
    ├── detection_pred
    ├── multilabel_grad_cam
    ├── multilabel_pred
    └── profile
```

## Configuration setting
- app_config.py
- app_config.yaml : backend(FastAPI) config
- front_config.py
- front_config.yaml : frontend(Streamlit) config

`.yaml` 파일에 있는 설정을 `.py` 파일로 읽어 전체 환경 설정 변수로 사용

## Frontend
- frontend.py : Streamlit app root

`Streamlit`의 기술적인 한계로 인해 Key 값등 여러 트릭이 활용됨

## Backend
- __main__.py : backend(FastAPI) Entrypoint
- main.py : FastAPI app root
- customrouter.py : main.py의 auto_include_router 함수와 함께 사용하여 `FastAPI`의 폴더 구조를 규격화 하고 통일함
- routes/ : customrouter 를 상속받아 실제 서비스되는 Endpoint가 정의됨
  - `명사1_명사2.py` -> `http://host/명사1/명사2` 의 구조로 접근 가능하게 설정됨
  - restful 한 설계를 따르도록 작성함

## Static
- static한 리소스를 제공하기 위한 경로
  - detection_pred : detection 요청으로 들어온 이미지와 예측 결과를 저장하는 경로
  - multilabel_grad_cam : multilabel classification 중 상세 정보 표시를 위해 grad cam을 저장하는 경로
  - multilabel_pred : multilabel classification 요청으로 들어온 이미지와 예측 결과를 저장하는 경로
  - profile : 메인 페이지의 팀원 소개를 위한 profile 사진 경로
  
