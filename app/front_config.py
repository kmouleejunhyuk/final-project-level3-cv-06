import yaml
from easydict import EasyDict as edict

import os

# entrypoint(__main__.py)로 진입시 경로 오류 해결을 위해 현재 디렉토리를 현재 파일의 경로로 변경
current_path = os.path.dirname(__file__)
os.chdir(current_path)

# read config
with open("./front_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config)