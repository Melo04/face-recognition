import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
import cv2
import os

st.set_page_config(page_title="Capture Faces", page_icon="📸")

st.markdown("# Capture Your Face Now 📸")

ROOT_DIR = ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))

FACE_DB = os.path.join(ROOT_DIR, "face_database")

if not os.path.exists(FACE_DB):
    os.mkdir(FACE_DB)

data_path = FACE_DB
file_db = os.path.join(FACE_DB, "face_db.csv")
file_history = os.path.join(FACE_DB, "face_history.csv")
COLS_INFO   = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(128)]

def initialize_data():
    if os.path.exists(os.path.join(data_path, file_db)):
        df = pd.read_csv(os.path.join(data_path, file_db))
    else:
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_vcsv(os.path.join(data_path, file_db), index=False)
    return df

def add_data_db(df_face_details):
    df = pd.read_csv(os.path.join(data_path, file_db))
    if not df.empty:
        df = pd.concat([df, df_face_details], ignore_index=False)
        df.drop_duplicates(keep='first', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(os.path.join(data_path, file_db), index=False)
        st.success("Image Uploaded Successfully")
    else:
        df_face_details.to_csv(os.path.join(data_path, file_db), index=False)
        st.success("Updated Data Successfully")

face_name = st.text_input('Input Your Name Below')
pic = st.selectbox('Select the mode you want to upload for', ('Upload a Picture', 'Capture Image via webcam'))

if pic == "Upload a Picture":
    img_file = st.file_uploader("Upload a picture", type=["png", "jpg", "jpeg"])

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)

elif pic == "Capture Image via webcam":
    img_file = st.camera_input("Capture Your Image")
    if img_file is not None:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)

if((img_file is not None) & (len(face_name) > 1) & st.button('Click to Save 💾')):
    image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with open(os.path.join(FACE_DB, f'{face_name}.jpg'), 'wb') as file:
        file.write(img_file.getbuffer())

    face_locations = face_recognition.face_locations(image_array)
    face_encodings = face_recognition.face_encodings(image_array, face_locations)

    df_new = pd.DataFrame(data=face_encodings, columns=COLS_ENCODE)
    df_new[COLS_INFO] = face_name
    df_new = df_new[COLS_INFO + COLS_ENCODE].copy()

    DB = initialize_data()
    add_data_db(df_new)