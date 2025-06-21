# kdeplot.py

import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import seaborn as sns

def generate_kde_plot(video_path, weights_path):
    # Load model
    model = YOLO(weights_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Could not read video.")
        return None
    height, width, _ = frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    x_coords, y_coords = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, class_id in zip(boxes, class_ids):
            if class_id == 2:  # Assuming class 2 is 'player'
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                x_pitch = (x_center / width) * 120
                y_pitch = (y_center / height) * 80

                x_coords.append(x_pitch)
                y_coords.append(y_pitch)

    cap.release()

    pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
    fig, ax = pitch.draw(figsize=(12, 8))

    sns.kdeplot(
        x=x_coords,
        y=y_coords,
        levels=15,
        fill=False,
        cmap="Reds",
        linewidths=1.5,
        ax=ax,
        thresh=0.01
    )

    ax.set_facecolor("black")
    plt.title("Player Position Density (KDE)", fontsize=20, color='white')
    plt.xlim(0, 120)
    plt.ylim(0, 80)
    plt.tight_layout()

    return fig
