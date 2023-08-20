import cv2
import os
import uuid
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import face_recognition

st.set_page_config(
    page_title="Face Recognition App",
    page_icon="🎭",
)

progress_text = "Loading..."
my_bar = st.progress(0, text=progress_text)
for percent_complete in range(100):
    time.sleep(0.05)
    my_bar.progress(percent_complete + 1, text=f"{percent_complete + 1}%  Loading...")
my_bar.empty()

st.write(
    """
    # Face Recognition Web App 🎭
    Hey there :wave:, welcome to my first Face Recognition web app made with Python, OpenCV as well as Streamlit.
    This app is capable of detecting faces via live detection web cam :camera_with_flash: or by uploading an image :open_file_folder:. 
    Head over to the sidebar to get started! :point_left:
    """
)

image = Image.open('assets/face.jpg')
st.image(image, use_column_width=True)

st.write('---')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DB = os.path.join(ROOT_DIR, "face_database")
FACE_HISTORY = os.path.join(ROOT_DIR, "face_history")

if not os.path.exists(FACE_HISTORY):
    os.mkdir(FACE_HISTORY)

if not os.path.exists(FACE_DB):
    os.mkdir(FACE_DB)

data_path = FACE_DB
file_db = 'face_db.csv'
file_history = 'face_history.csv'
COLS_INFO   = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(128)]

def initialize_data():
    if os.path.exists(os.path.join(data_path, file_db)):
        df = pd.read_csv(os.path.join(data_path, file_db))
    else:
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_csv(os.path.join(data_path, file_db), index=False)
    return df

st.header("Live Webcam Face Recognition :camera_with_flash:")
st.write(
    """
	Enable your webcam to start the live detection.
    Make sure to upload your image to the database first before starting the live detection.
	"""
)
face_id = uuid.uuid1()
img_file = st.camera_input("")

if img_file is not None:
    bytes_data = img_file.getvalue()
    image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with open(os.path.join(FACE_HISTORY, f'{face_id}.jpg'), 'wb') as file:
        maxi = 0
        rois = []
        file.write(img_file.getbuffer())
        face_locations = face_recognition.face_locations(image_array)
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            rois.append(image_array[top:bottom, left:right].copy())

        maxi = len(face_locations)
        if maxi > 0:
            st.success(f"Image Captured Successfully, {maxi} face detected")
            face_idxs = [i for i in range(maxi)]
            flag = False

            if st.button('Click to Detect Faces'):
                dataframe_new = pd.DataFrame()
                                                    
                for face_idx in face_idxs:
                    database_data = initialize_data()
                    face_encodings = database_data[COLS_ENCODE].values
                    dataframe = database_data[COLS_INFO]
                    faces = face_recognition.face_encodings(rois[face_idx])

                    if len(faces) < 1:
                        st.error(f'Please try again. No face detected for {face_idx}')
                    else:
                        dataframe['distance'] = face_recognition.face_distance(face_encodings, faces)
                        name = ""
                        distance = 1e5+9
                        for i in range(len(dataframe['distance'])):
                            if dataframe['distance'][i] < distance:
                                distance = dataframe['distance'][i]
                                name = dataframe['Name'][i]
                        # st.write(dataframe['Name'], dataframe['distance'])
                        # st.write(name, distance)
                        (top, right, bottom, left) = (face_locations[face_idx])
                        rois.append(image_array[top:bottom, left:right].copy())
                        cv2.rectangle(image_array, (left, top), (right, bottom), (255,0,0), 2)
                        cv2.rectangle(image_array, (left, bottom), (right, bottom), (255,0,0), cv2.FILLED)
                        confidence = (1.0 - distance) * 100
                        cv2.putText(image_array, f"{name}, {confidence:.0f}%", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 1)
                        
                        st.success(f"Face Detected: {name}, Confidence level: {confidence:.0f}%")
                        flag = True

                if flag:
                    st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            st.error("No face detected, please try again")

st.write('---')
st.header("List Of Celebrities That Can Be Detected 🎭")

celebrities = [
    {"name": "Chris Hemsworth", "image": "assets/Chris Hemsworth.jpg"},
    {"name": "Robert Downey Jr.", "image": "assets/Robert Downey Jr.jpeg"},
    {"name": "Tom Holland", "image": "assets/Tom Holland.jpg"},
    {"name": "Tom Hiddleston", "image": "assets/Tom Hiddleston.jpg"},
    {"name": "Scarlett Johansson", "image": "assets/Scarlet Johansson.jpg"},
    {"name": "Chris Evans", "image": "assets/Chris Evans.jpg"},
    {"name": "Elon Musk", "image": "assets/Elon Mask.jpg"},
    {"name": "Chris Pratt", "image": "assets/Chris Pratt.jpg"},
    {"name": "Zendaya", "image": "assets/Zendaya.jpg"},
]

columns_num = 3
rows_num = len(celebrities) // columns_num + (len(celebrities) % columns_num > 0)

grid_cols = st.columns(columns_num)

for i, celebrity in enumerate(celebrities):
    with grid_cols[i % columns_num]:
        st.image(celebrity["image"], caption=celebrity["name"], use_column_width=True)