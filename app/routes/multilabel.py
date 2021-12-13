from app.customrouter import fileRouter
from fastapi.responses import RedirectResponse

from app.app_config import config as CONFIG

MODEL = CONFIG.multilabel_model
DEVICE = CONFIG.device
LABELS = CONFIG.classes


router = fileRouter(prefix=__file__)

@router.get("/", response_class=RedirectResponse)
def root():
    return "./pred"

@router.get("/model")
def model():
    return MODEL.name

@router.get("/device")
def device():
    return DEVICE

@router.get("/labels")
def labels():
    return LABELS