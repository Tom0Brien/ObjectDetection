import cv2
import numpy as np

# Define the URL of the video stream
url = 'udp://192.168.0.2:1234'

# Load the pre-trained YOLOv3 model and configure the network
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define the minimum confidence threshold for human detection
conf_threshold = 0.5

# Define the maximum overlap threshold for non-maximum suppression
iou_threshold = 0.4

# Open the video stream
cap = cv2.VideoCapture(url)

# Check if the stream was successfully opened
if not cap.isOpened():
    print("Failed to open the video stream")
    exit()

# Read the frames from the stream
while True:
    # Read a frame from the stream
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Failed to read a frame from the stream")
        break

    # Detect humans in the frame using the YOLOv3 model
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers)

    # Process the model outputs to extract human detections
    humans = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > conf_threshold:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x = center_x - width//2
                y = center_y - height//2
                humans.append([x, y, width, height, confidence])

    # Apply non-maximum suppression to remove overlapping bounding boxes
    if len(humans) > 0:
        boxes = np.array(humans)[:, :4]
        confidences = np.array(humans)[:, 4]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
        humans = [humans[i] for i in indices.flatten()]

    # Draw the human detections on the frame
    for x, y, w, h, conf in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Human", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show the frame in a window
    cv2.imshow('Video Stream', frame)

    # Wait for a key event and check if 'q' was pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows