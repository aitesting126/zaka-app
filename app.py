import json
# import boto3
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
from PIL import Image
import io
import requests

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
lottie_coding = load_lottiefile("stuff/upload.json")


st.set_page_config(page_title="Segmentation pipeline", layout="wide")


# AWS_REGION = st.secrets["AWS_REGION"]
# AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
# AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

# client = boto3.client(
#     'sagemaker-runtime', 
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name=AWS_REGION
# )


with st.container():
    st.title("Segmenation pipeline")
    st.write("##")
    st.header("Introduction")
    st.write(
        """
        A segmentation pipeline is a sequence of processes that divides input data into meaningful segments, often used in tasks like image analysis, object recognition, and data labeling. 
        This pipeline is crucial in various domains, facilitating accurate understanding and interpretation of complex data.

        This is a project built by 2 ML Engineers students of Zaka:
        - Bohdan Stadnyk: with 10 years of programming experience
        - Emile Boulos: Telecom engineering, and pianist for 13 years

        This project woudn't have been done without the effort of my collegue Bohdan, and the Zaka AI comunity.
        """
    )


with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Project")
        st.write("##")
        st.write("##")
        st.write(
            """
            The project aims to create an effective segmentation pipeline that can be applied to various domains while providing accurate and reliable segmentation results for different types of images and datasets. Dataset is called VOC2012.

            The model is trained on detecting 20 object classes:  
            - person
            - bird, cat, cow, dog, horse, sheep
            - Aeroplan, bicycle, boat, bus, car, motorbike, train
            - bottle, chair, dining table, potted plant, sofa, tv/monitor

            Our solution aims at masking a detected object of the stated above.
            
            """
        )
        st.write("##")
        st.markdown(
            """
                <p style='color: lightcoral;'>
                    In order to upload your image, you can drag and drop it to the gray section area, or press on "Browse files" on the right side
                </p>
            """,
            unsafe_allow_html=True
        )
        
    with right_column:
        st.header("Upload your image")
        # Create columns for layout
        left_column, middle_column, right_column = st.columns([1, 3, 1])

        # Place the animation in the middle column
        with middle_column:
            st_lottie(
                lottie_coding,
                speed=0.9,
                loop=True,
                width=350,
                height=350
            )
            # if st.button("Browse"):
            #     # Perform the browsing action here
            #     browse_action()

        uploaded_file = st.file_uploader("", type=["png", 'jpg'])

        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()

            files = {'file': ('filename.jpeg', bytes_data)}
            response = requests.post("https://f50a-78-110-74-199.ngrok-free.app", files=files)


            image = Image.open(io.BytesIO(response.content))

            st.image(bytes_data, caption="Original Image", use_column_width=True)
            st.image(image, caption="Processed Image", use_column_width=True)



with st.container():
    st.write("---")
    st.header("Why VOC2012")
    st.write("##")
    st.write(
        """
        VOC2012 is a dataset that can be used in classification/detection, segmentation, action classification and person layout taster. 
        
        We used this dataset in our project in order, which contains the original images and their masks, in order to train and deploy the model. 
        
        The 2012 dataset contains images from 2008-2011 for which additional segmentations have been prepared. As in previous years the assignment to training/test sets has been maintained. The total number of images with segmentation has been increased from 7,062 to 9,993.
        
        """
    )
    st.write("[Link to the dataset >](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data)")
    st.write("---")
