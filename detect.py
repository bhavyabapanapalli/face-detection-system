
import cv2

print("Face Detection System")
print("OpenCV version:", cv2.__version__)
print("Setup successful!")



import cv2

image = cv2.imread("sample_images/person.jpg")

if image is None:
    print("Error: Image not found!")
else:
    print("Image loaded successfully!")

    cv2.imshow("My Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




import cv2

# Step 1: Read the image
image = cv2.imread("sample_images/person.jpg")

if image is None:
    print("Error: Image not found!")
    raise SystemExit

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Load the face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

# Step 4: Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Step 5: Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

# Step 6: Print results
print("Faces detected:", len(faces))

# Step 7: Display image
cv2.imshow("Face Detection System", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Save the detected image
output_path = "sample_images/detected_faces.jpg"
cv2.imwrite(output_path, image)

print("Detected image saved at:", output_path)

