# Ai_football_analyzer


**AI Football Analyzer** is a real-time, intelligent football analysis system designed to analyze match visuals from **images** or **videos**. This tool empowers users to perform **detection**, **segmentation**, and **pose estimation** using cutting-edge deep learning models — all through an intuitive **Streamlit interface**.

---

## 🌟 Key Features

### 📸 Image Processing
- Upload any football match image.
- Choose between:
  - **Object Detection**: Identify players, goalkeeper, ball.
  - **Semantic Segmentation**: Differentiate regions (players, field, goalpost).
  - **Pose Estimation**: Extract body landmarks of players.

### 🎥 Video Analysis
- Upload videos (ideally 30–60 seconds, bird’s-eye view).
- Real-time inference using **OpenCV**.
- Frame-by-frame analysis (every 30 frames) including:
  - **Live Commentary** powered by GPT.
  - **Player Detection & Pose Estimation**.
  - **KDE Heatmaps** for movement distribution.

### 📊 Field Analytics
- Confidence adjustment trigger for bbox frames
- Generate **KDE (Kernel Density Estimation) plots** of player density.
- Visualize tactical spread and player activity on the pitch.

---

## 🚀 Technical Architecture

| Module            | Technology                               |
|------------------|-------------------------------------------|
| Model Backbone    | **Merged YOLOv8 + YOLOv11** (for Detection, Pose, Segmentation) |
| Interface         | **Streamlit**                            |
| Real-time Vision  | **OpenCV**                               |
| Audio Commentary  | **GPT + TTS (pyttsx3 or Coqui)**         |
| Analytics Plots   | **Seaborn, Matplotlib**                  |

---


## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### A) Prerequisites

* **Git:** You must have Git installed to clone the repository. You can download it from [git-scm.com](https://git-scm.com/).
* **Python 3.10:** This project requires Python version 3.10. You can download it from [python.org](https://www.python.org/downloads/release/python-3100/).
* **NVIDIA GPU with CUDA 12.1 compatible drivers:** The specified PyTorch version is built for CUDA 12.1. Ensure your GPU drivers are up-to-date.

### B) Installation & Setup

Follow these steps to set up your local development environment.

1.  **Clone the GitHub Repository**

    Open your terminal or command prompt and use the following command to clone the repository. 

    ```bash
    git clone https://github.com/Om-mhaismale/Ai_football_analyzer.git
    cd Ai_football_analyzer
    ```

2.  **Create and Activate a Python 3.10 Virtual Environment**

    It's highly recommended to use a virtual environment to manage project-specific dependencies.

    * **On macOS and Linux:**
        ```bash
        python3.10 -m venv venv
        source venv/bin/activate
        ```

    * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    After activation, your terminal prompt should be prefixed with `(venv)`.

3.  **Install GPU-Compatible PyTorch**

    First, ensure any existing versions of PyTorch are uninstalled to avoid conflicts. Then, install the version compatible with CUDA 12.1.

    ```bash
    # Uninstall any old versions first
    pip uninstall torch torchvision torchaudio -y

    # Install PyTorch for CUDA 12.1
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

4.  **Verify PyTorch Installation**

    You can quickly check if PyTorch was installed correctly and can see your GPU.

    ```bash
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device name: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No CUDA device found')"
    ```

5.  **Install Other Requirements**

    ```bash
    pip install -r requirements.txt
    ```

## C) Usage

* To run the application, use the following command in your terminal:


    ```bash
   streamlit run main.py
    ```


## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

