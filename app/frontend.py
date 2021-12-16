import io

import numpy as np
import requests
import streamlit as st
from PIL import Image


def main():
    st.set_page_config(
        page_title="X-Ray Baggage Scanner 자동 검출 솔루션",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # home_button = st.sidebar.button("Home")
    model_radio = st.sidebar.radio("Model Select", ("Not Selected", "Multi-label Classification", "Object detection"))   
    if model_radio == "Multi-label Classification":
        model_response = requests.get("http://203.252.79.155:8002/multilabel/model")
        st.sidebar.warning(model_response.json())
    elif model_radio == "Object detection":
        model_response = requests.get("http://203.252.79.155:8002/detection/model")
        st.sidebar.warning(model_response.json())
    
    st.header("X-Ray Baggage Scanner 자동 검출 솔루션")

    placeholder = st.empty()

    with placeholder.container(): # 6개 st line
        st.markdown("____")
        st.subheader("프로젝트 소개")
        st.write("주제 : X-Ray Baggage Scanner 자동 검출 솔루션  \n 설명 : 공항의 수화물에 포함된 유해물품(흉기, 화기류 등)을 CV기반 솔루션으로 검출  \n task 1 : Multi-label Classification  \n task 2 : Object Detection Model")
        st.markdown("____")
        st.subheader("팀원 소개")
        name_route = '/home/jhoh/finalproject/app/static/profile/'
        member_images = ['jiwoo.png', 'jiyun.png', 'jiyou.jpeg', 'naeun.jpg', 'jaehwan.jpg', 'junhyuk.png', 'kyungjae.png']
        member_route = [name_route+member for member in member_images]
        st.image(member_route, width=100, caption=["jiwoo", "jiyun", "jiyou", "naeun", "jaehwan", "junhyuk", "kyungjae"])

    mode_radio = st.sidebar.radio("Mode Select", ("Not Selected", "New Image", "Predicted Image"))
    if mode_radio == "New Image":
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png"])

        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            files = [
                ('files', (uploaded_file.name, image_bytes, uploaded_file.type))
            ]

            if model_radio == 'Multi-label Classification':
                placeholder.empty()
                with placeholder.container():
                    st.markdown("____")
                    classifying_msg = st.warning("Classifying...")

                    cls_response = requests.post("http://203.252.79.155:8002/multilabel/pred/", files=files)
                    st.write(f'image similarity : {cls_response.json()[1]}')
                    if float(cls_response.json()[1]) < 0.5:
                        st.warning("It is not a X-ray image!")
                        classifying_msg.empty()
                        st.write("")
                        st.write("")
                    else:
                        st.write(f'labels : {cls_response.json()[0]}')
                        grad_cam = Image.fromarray(np.array(cls_response.json()[2]).astype('uint8'))
                        st.image(grad_cam, caption="Uploaded Image")

                        classifying_msg.empty()
                        st.success("Classificated!!")

            elif model_radio == 'Object detection':
                placeholder.empty()
                with placeholder.container():
                    st.markdown("____")
                    detecting_msg = st.warning("Detecting...")

                    detect_response = requests.post("http://203.252.79.155:8002/detection/pred/", files=files)
                    img_arr = np.array(detect_response.json())
                    detect_image = Image.fromarray(img_arr.astype('uint8'))
                    st.write("")
                    st.image(detect_image, caption="Detected image")

                    detecting_msg.empty()
                    st.success("Detected!!")
                    st.write("")

    elif mode_radio == "Predicted Image":
        file_response = requests.get("http://203.252.79.155:8002/multilabel/pred/")
        file_select = st.sidebar.selectbox("Images", file_response.json()) # key
        
        if file_select != 'None':
            if model_radio == 'Multi-label Classification':
                placeholder.empty()
                with placeholder.container():
                    result = requests.get(f"http://203.252.79.155:8002/multilabel/pred/{file_select}")
                    
                    st.write(f'labels : {result.json()[1]}')
                    
                    result_img = Image.open(result.json()[0])
                    st.image(result_img, caption="Result Image")
                    st.success("Success load!!")

main()
