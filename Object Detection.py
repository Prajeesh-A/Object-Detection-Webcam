import cv2
import numpy as np

# Load the pre-trained deep learning model
prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Define class labels for objects the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "monitor" , "mouse"]

# Open a connection to the webcam
cap = cv2.VideoCapture("http://192.168.1.35:4747/video")  # Adjust IP from DroidCam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for model compatibility
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Perform object detection
    net.setInput(blob)
    detections = net.forward()

    # Loop through detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])  # Class index
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"{label}: {confidence * 100:.2f}%"
            cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Real-time Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
