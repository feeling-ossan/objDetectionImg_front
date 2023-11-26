import streamlit as st
import requests
import numpy as np
import cv2
from io import BytesIO

# バックエンドサーバURL
# ローカル
# buckend_url = 'http://127.0.0.1:8000/objDetection'
# Render
buckend_url = 'https://objdetectionimg-back.onrender.com/objDetection'

st.title("Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=['png','jpg'])

if uploaded_file is not None:
    # アップロードデータのread
    readData = uploaded_file.read()

    # アップロード画像の表示
    st.write('## Object Detection Before')
    file_bytes = np.asarray(bytearray(readData), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels='BGR', caption='Uploaded Image', use_column_width=True)

    # バックエンド側へrequest
    files = {'file': ('filename', readData, 'image/jpeg')}
    response = requests.post(buckend_url, files=files, timeout=60)

    # response受信
    if response.status_code == 200:
        # 結果をバッファに書き込み
        image_buffer = BytesIO(response.content)
        
        # バッファから画像を読み込む
        np_image = np.frombuffer(image_buffer.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # 物体検知結果を表示
        st.write('## Object Detection After')
        st.image(image, channels='BGR', caption="Object Detection Result", use_column_width=True)

    else:
        st.error(f"Error: {response.status_code}")
