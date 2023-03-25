import cv2
import numpy as np

# Define the URL of the video stream
url = 'udp://localhost:1234'

# Load the pre-trained YOLOv3 model and configure the network
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define the minimum confidence threshold for detection
conf_threshold = 0.5

# Define the maximum overlap threshold for non-maximum suppression
iou_threshold = 0.4

# Define the labels for the classes
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define the colors for the bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

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

    # Detect objects in the frame using the YOLOv3 model
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (128, 128), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers)

    # Process the model outputs to extract detections
    detections = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x = center_x - width//2
                y = center_y - height//2
                detections.append([x, y, width, height, confidence, class_id])

    # Apply non-maximum suppression to remove overlapping bounding boxes
    if len(detections) > 0:
        boxes = np.array(detections)[:, :4]
        confidences = np.array(detections)[:, 4]
        class_ids = np.array(detections)[:, 5]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
        detections = [detections[i] for i in indices.flatten()]

    # Draw the detections on the frame
    for x, y, w, h, conf, class_id in detections:
        label = f'{classes[class_id]}: {conf:.2f}'
        color = colors[class_id]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame in a window
    cv2.imshow('Video Stream', frame)

# Wait for a key event and check if 'q' was pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
