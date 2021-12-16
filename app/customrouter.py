from fastapi import APIRouter
import os
import re

class fileRouter(APIRouter):
    '''
    파일명 기반으로 라우터 생성
    
    Args:
        prefix : kwargs로 설정 필수, __file__ 입력 권장
    '''
    def __init__(self, *args, **kwargs):
        # if os.path.isfile(prefix := kwargs.get("prefix", None)):
        # os.path.isfile 테스트 환경문제로 생략, 오류 확인 신경쓸것
        if (prefix := kwargs.get("prefix", None)):
            prefix, ext = os.path.splitext(os.path.basename(prefix))
            prefix = re.sub("_", "/", prefix)
            kwargs["prefix"] = "/" + prefix
            super().__init__(*args, **kwargs)
        else:
            raise Exception(f"prefix가 지정되지 않았거나 파일 경로(__file__)가 아닙니다. \n  prefix : [{prefix}]")