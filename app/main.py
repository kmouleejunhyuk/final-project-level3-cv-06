from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import importlib
import traceback
import os

from app.app_config import config as CONFIG


def auto_include_router(app, dir_path, lib_path):
    '''
    root 내부의 경로 일괄 include

    현재 1 depth를 초과하는 디렉토리 인식 불가
    필요시 내부 디렉토리 재귀적 호출 방식으로 개선 필요
    (프로젝트 규모가 작아 아래 방식이 유리할거라고 생각됨)

    Args :
        app: FastAPI 객체
        dir_path: 디렉토리 탐색을 위한 폴더 경로 ex) routes/abc/def
        lib_path: 모듈 임포트를 위한 경로 ex) app.route.multilabel

    Router setting rule :
        URL/a/b -> routes/a_b.py
        router = APIRouter(prefix="/a/b")
            .
            상세 구현
            .
    '''
    paths = os.listdir(dir_path)
    for path in paths:
        if path[0] != "_":
            try:
                name, ext = os.path.splitext(path)
                module = importlib.import_module(lib_path + name)
                app.include_router(module.router)
                print(f"[Success] include_router : {name}")
            except Exception as e:
                print(f"[Error] include_router : {path}")
                traceback.print_exc()
            

app = FastAPI()
app.mount("/" + CONFIG.static.url, StaticFiles(directory=CONFIG.static.directiory), name=CONFIG.static.directiory)
auto_include_router(app, CONFIG.router.dir_path, CONFIG.router.lib_path)

@app.get("/")
def root():
    return {"hello" : "world"}
    
