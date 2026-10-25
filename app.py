
import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Face Detection System",
    page_icon="🔍",
    layout="centered"
)

st.title("Face Detection System")
st.write("Upload an image to detect faces using OpenCV.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.getvalue()),
        dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read this image.")
    else:
        # Load face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        # Convert image to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

        # Display results
        result = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.success(f"Faces detected: {len(faces)}")
        st.image(
            result,
            caption="Face Detection Result",
            use_container_width=True
        )

        # Download processed image
        success, encoded = cv2.imencode(".jpg", image)

        if success:
            st.download_button(
                "Download Detected Image",
                data=encoded.tobytes(),
                file_name="detected_faces.jpg",
                mime="image/jpeg"
            )
