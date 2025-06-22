#Import All the Required Libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO 
from PIL import Image
from kdeplot import generate_kde_plot  
from extract_positions import extract_positions, count_detected_objects
from commentary_generator import generate_commentary
from text_to_audio import generate_audio


#Get the absolute path of the current file
FILE = Path(__file__).resolve()

#Get the parent directory of the current file
ROOT = FILE.parent

#Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

#Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

#Sources
IMAGE = 'Image'
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, VIDEO]

#Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'image1.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'detected_img.png'

#Videos Config
VIDEO_DIR = ROOT/'videos'
VIDEOS_DICT = {
    'video 1': VIDEO_DIR/'video1.mp4',
    'video 2': VIDEO_DIR/'video2.mp4'
}

#Model Configurations
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'best.pt'

#In case of your custom model
#DETECTION_MODEL = MODEL_DIR/'custom_model_weight.pt'

SEGMENTATION_MODEL  = MODEL_DIR/'yolo11n-seg.pt'

POSE_ESTIMATION_MODEL = MODEL_DIR/'yolo11n-pose.pt'

#Page Layout
st.set_page_config(
    page_title = "lets analyze football matches",
    page_icon = "⚽"
)

#Header
st.header("AI Football Analyses")
st.markdown(
    "This application performs best with videos captured from a bird's-eye (top-down) camera angle and with a duration between 30 and 60 seconds."
)

#SideBar
st.sidebar.header("Model Configurations")

#Choose Model: Detection, Segmentation or Pose Estimation
model_type = st.sidebar.radio("Task", ["Detection", "Segmentation", "Pose Estimation"])

#Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40))/100

#Selecting Detection, Segmentation, Pose Estimation Model
if model_type == 'Detection':
    model_path = Path(DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(SEGMENTATION_MODEL)
elif model_type ==  'Pose Estimation':
    model_path = Path(POSE_ESTIMATION_MODEL)

#Load the YOLO Model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the sepcified path: {model_path}")
    st.error(e)

#Image / Video Configuration
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", SOURCES_LIST
)

source_image = None
if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader(
        "Choose an Image....", type = ("jpg", "png", "jpeg", "bmp", "webp")
    )
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, caption = "Default Image", use_container_width=True)
            else:
                uploaded_image  = Image.open(source_image)
                st.image(source_image, caption = "Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error("Error Occurred While Opening the Image")
            st.error(e)
    with col2:
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption = "Detected Image", use_container_width=True)
            else:
                detect_btn = st.sidebar.button("Detect Objects")
                commentary_btn = st.sidebar.button("Generate Commentary")
                if detect_btn:
                    result = model.predict(uploaded_image, conf=confidence_value)
                    boxes = result[0].boxes
                    # st.write("Detected class IDs:", [int(box.cls) for box in boxes])
                    result_plotted = result[0].plot()[:,:,::-1]
                    st.image(result_plotted, caption="Detected Image", use_container_width=True)
                    st.session_state['last_image_result'] = result  # Save for commentary

                    # Show detection summary
                    positions = extract_positions(result)
                    counts = count_detected_objects(result)
                    st.markdown("**Detection Summary:**")
                    st.write(f"Players: {counts['players']}")
                    st.write(f"Goalkeepers: {counts['goalkeepers']}")
                    st.write(f"Main Referees: {counts['main_referees']}")
                    st.write(f"Side Referees: {counts['side_referees']}")
                    st.write(f"Staff Members: {counts['staff_members']}")
                    st.write(f"Ball detected: {'Yes' if counts['ball'] > 0 else 'No'}")

                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as e:
                        st.error(e)
                if commentary_btn:
                    result = st.session_state.get('last_image_result', None)
                    if result is not None:
                        positions = extract_positions(result)
                        commentary = generate_commentary(positions)
                        audio_file = generate_audio(commentary)
                        st.write("AI Commentary:")
                        st.write(commentary)
                        st.audio(audio_file, format='audio/wav')
                    else:
                        st.warning("Please run detection first.")
        except Exception as e:
            st.error("Error Occurred While Opening the Image")
            st.error(e)

    if source_image is not None:
        # Reset detection result if a new image is uploaded
        if 'last_image_name' not in st.session_state or st.session_state['last_image_name'] != source_image.name:
            st.session_state['last_image_result'] = None
            st.session_state['last_image_name'] = source_image.name

elif source_radio == VIDEO:
    uploaded_video = st.sidebar.file_uploader(
        "Upload a Video...", type=("mp4", "avi", "mov", "mkv")
    )
    if uploaded_video is not None:
        st.video(uploaded_video)
        temp_video_path = "temp_uploaded_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        detect_btn = st.sidebar.button("Detect Video Objects (Uploaded)")
        commentary_btn = st.sidebar.button("Generate Commentary (Uploaded Video)")
        kde_btn = st.sidebar.button("Generate KDE Plot (Uploaded Video)")  # <-- Added to sidebar
        if detect_btn:
            try:
                video_cap = cv2.VideoCapture(temp_video_path)
                st_frame = st.empty()
                st.session_state['video_results'] = []
                while video_cap.isOpened():
                    success, image = video_cap.read()
                    if success:
                        image = cv2.resize(image, (720, int(720 * (9/16))))
                        result = model.predict(image, conf=confidence_value)
                        result_plotted = result[0].plot()
                        st_frame.image(result_plotted, caption="Detected Video",
                                       channels="BGR",
                                       use_container_width=True)
                        st.session_state['video_results'].append(result)
                    else:
                        video_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error Loading Uploaded Video: " + str(e))
        if commentary_btn:
            results = st.session_state.get('video_results', [])
            if results:
                for idx, result in enumerate(results):
                    if idx == 0:
                        continue  # Skip frame 0
                    if idx % 30 == 0:
                        positions = extract_positions(result)
                        positions["frame_id"] = idx  # ✅ Add this line
                        commentary = generate_commentary(positions)
                        audio_file = generate_audio(commentary)
                        st.write(f"AI Commentary (frame {idx}):")
                        st.write(commentary)
                        st.audio(audio_file, format='audio/wav')
            else:
                st.warning("Please run detection first.")

        if kde_btn:
            with st.spinner("Generating KDE plot..."):
                fig = generate_kde_plot(temp_video_path, str(model_path))
                if fig:
                    st.pyplot(fig)
                else:
                    st.error("Failed to generate KDE plot.")
    else:
        video_path = str(next(iter(VIDEOS_DICT.values())))
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            if video_bytes:
                st.video(video_bytes)
            detect_btn = st.sidebar.button("Detect Video Objects")
            commentary_btn = st.sidebar.button("Generate Commentary (Sample Video)")
            kde_btn = st.sidebar.button("Generate KDE Plot (Sample Video)")  # <-- Added to sidebar
            if detect_btn:
                try:
                    video_cap = cv2.VideoCapture(video_path)
                    st_frame = st.empty()
                    st.session_state['sample_video_results'] = []
                    while video_cap.isOpened():
                        success, image = video_cap.read()
                        if success:
                            image = cv2.resize(image, (720, int(720 * (9/16))))
                            result = model.predict(image, conf = confidence_value)
                            result_plotted = result[0].plot()
                            st_frame.image(result_plotted, caption = "Detected Video",
                                           channels = "BGR",
                                           use_container_width=True)
                            st.session_state['sample_video_results'].append(result)
                        else:
                            video_cap.release()
                            break
                except Exception as e:
                    st.sidebar.error("Error Loading Video"+str(e))
            if commentary_btn:
                results = st.session_state.get('sample_video_results', [])
                if results:
                    for idx, result in enumerate(results):
                        if idx % 30 == 0:
                            positions = extract_positions(result)
                            positions["frame_id"] = idx  # ✅ Add this line
                            commentary = generate_commentary(positions)
                            audio_file = generate_audio(commentary)
                            st.write(f"AI Commentary (frame {idx}):")
                            st.write(commentary)
                            st.audio(audio_file, format='audio/wav')
                else:
                    st.warning("Please run detection first.")

            if kde_btn:
                with st.spinner("Generating KDE plot..."):
                    fig = generate_kde_plot(video_path, str(model_path))
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.error("Failed to generate KDE plot.")