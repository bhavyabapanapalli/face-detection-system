
# Face Detection System

A Computer Vision project built using Python, OpenCV, NumPy, and Streamlit.

## Features
- Detect faces in uploaded images
- Draw bounding boxes around detected faces
- Count multiple faces
- Download processed images
- Real-time webcam face detection

## Technologies Used
- Python
- OpenCV
- NumPy
- Streamlit

## Project Structure
- `detect.py` — Image face detection
- `webcam.py` — Live webcam face detection
- `app.py` — Streamlit web application
- `requirements.txt` — Project dependencies
- `sample_images/` — Sample and output images

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Web Application

```bash
streamlit run app.py
```

## Run Webcam Detection

```bash
python webcam.py
```

Press ESC to exit webcam detection.

## Note
This project detects faces but does not identify a person's identity.
