import io

import albumentations as A
import numpy as np
import requests
import streamlit as st
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image


def main():
    st.set_page_config(
        page_title="X-Ray Baggage Scanner 자동 검출 솔루션",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.title("X-Ray Baggage Scanner 자동 검출 솔루션")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg","png"])
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        st.image(image, caption="Uploaded Image")
        classifying_msg = st.warning("Classifying...")

        files = [
            ('files', (uploaded_file.name, image_bytes, uploaded_file.type))
        ]

        cls_response = requests.post("http://203.252.79.155:8002/multilabel/pred/", files=files)
        st.write(f'labels : {cls_response.json()}')
        classifying_msg.empty()

        warning_message = st.warning("위해물품의 위치를 추적하시겠습니까?")
        OD_yes_button = st.button("네")
        OD_no_button = st.button("아니오")
        if OD_yes_button:
            warning_message.empty()
            image_bytes = uploaded_file.getvalue()

            detecting_msg = st.warning("Detecting...")

            detect_response = requests.post("http://203.252.79.155:8002/detection/pred/", files=files)
            img_arr = np.array(detect_response.json())
            detect_image = Image.fromarray(img_arr.astype('uint8'))
            st.image(detect_image, caption="Detected image")

            detecting_msg.empty()

            # OD_yes_button.empty()
            # OD_no_button.empty()
        
        elif OD_no_button:
            # OD_yes_button.empty()
            # OD_no_button.empty()
            pass

    # add_selectbox = st.sidebar.selectbox("왼쪽 사이드바 Select Box", ("A", "B", "C"))

    st.text(" ")
    
    st.title("프로젝트 소개")
    st.text("주제 : X-Ray Baggage Scanner 자동 검출 솔루션")
    st.text("설명 : 공항의 수화물에 포함된 유해물품(흉기, 화기류 등)을 CV기반 솔루션으로 검출")
    st.text("task 1 : Multi-label Classification")
    st.text("task 2 : Object Detection Model")
    
    
    # st.title("팀원 소개")
    # naeun_route = '/opt/ml/finalproject/detection/naeun.jpeg'
    # member_images = [naeun_route, naeun_route, naeun_route, naeun_route, naeun_route, naeun_route, naeun_route]
    # st.image(member_images, width=100,caption=["naeun", "naeun", "naeun", "naeun", "naeun", "naeun", "naeun"])
        
main()
