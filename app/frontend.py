from io import BytesIO
from random import randint
import requests
from PIL import Image

import streamlit as st

from app.front_config import config as CONFIG

BACK_ADDRESS = "http://" + CONFIG.backend_ip
BACK_PORT = str(CONFIG.backend_port)
MODELS = CONFIG.MODELS
MEMBER = CONFIG.MEMBER


def main():
    st.set_page_config(
        page_title="X-Ray Baggage Scanner 자동 검출 솔루션",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    if 'key' not in st.session_state:
        st.session_state.key = str(randint(1000, 100000000))

    ### side bar

    # 홈버튼
    home_button = st.sidebar.button("Home")

    # 모델 선택
    model_type = st.sidebar.radio("Model Select", MODELS)
    model_response = requests.get(BACK_ADDRESS + ":" + BACK_PORT + "/" + model_type + "/model")
    device_response = requests.get(BACK_ADDRESS + ":" + BACK_PORT + "/" + model_type + "/device")
    labels_response = requests.get(BACK_ADDRESS + ":" + BACK_PORT + "/" + model_type + "/labels")
    st.sidebar.info("[" + device_response.json() + "] " + model_response.json()  
                    + ", total " + str(len(labels_response.json())) + " categories")

    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=CONFIG.input_type,  key=st.session_state.key)

    # 과거 이미지 선택
    file_response = requests.get(BACK_ADDRESS + ":" + BACK_PORT + "/" + model_type + "/pred")
    pred_images = list(file_response.json().keys())
    pred_images.append("Select Image")
    pred_images.sort(reverse=True)
    idx = 1 if st.session_state.pop('new', None) else 0 # new가 true면 select box 생성시 가장 최근 이미지 자동 선택
    file_select = st.sidebar.selectbox("Select Images", pred_images, index=idx, key=st.session_state.key)
    
    # 홈버튼 누르면 세션 초기화 후 새로고침
    if home_button and 'key' in st.session_state.keys():
        st.session_state.pop('key')
        st.experimental_rerun()

    ### main page

    # Header
    st.header("X-Ray Baggage Scanner 자동 검출 솔루션")
    st.markdown("____")

    if uploaded_file:
        # new image
        image_bytes = uploaded_file.getvalue()
        files = [('files', (uploaded_file.name, image_bytes, uploaded_file.type))]
        with st.spinner("Classifying..."):
            pred_response = requests.post(BACK_ADDRESS + ":" + BACK_PORT + "/" + model_type + "/pred", files=files)

        st.session_state.pop('key')
        st.session_state.new = True # new가 true면 select box 생성시 가장 최근 이미지 자동 선택
        st.experimental_rerun()
        
    elif file_select != "Select Image":
        # pred
        st.subheader("Prediction")
        pred_response = requests.get(BACK_ADDRESS + ":" + BACK_PORT + "/" + model_type + "/pred/" + file_select)
        image_response = requests.get(BACK_ADDRESS + ":" + BACK_PORT + "/" + model_type + "/pred/grad/" + file_select)
        st.write(f'labels : {pred_response.json()}')
        image = Image.open(BytesIO(image_response.content)) 
        st.image(image, caption="Result Image")
        st.success("Success load!!")
        
    else:
        # home
        st.subheader("프로젝트 소개")        
        st.write("""
                주제 : X-Ray Baggage Scanner 자동 검출 솔루션  
                설명 : 공항의 수화물에 포함된 유해물품(흉기, 화기류 등)을 CV기반 솔루션으로 검출  
                task 1 : Multi-label Classification  
                task 2 : Object Detection Model
                """)
        st.markdown("____")
        st.subheader("팀원 소개")

        member_image_path = MEMBER.member_image_path
        member_images = []
        member_captions = []
        for member in MEMBER.members:
            member_images.append(member_image_path+member.image)
            member_captions.append(member.name)
        st.image(member_images, width=100, caption=member_captions)


main()
