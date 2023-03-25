# Object Detection using YOLOv3

This script uses the YOLOv3 object detection model to detect objects in a video stream. It can be used with any video stream, as long as the URL of the stream is specified in the script.

## Requirements

To run this script, you will need to have the following dependencies installed:

- OpenCV
- NumPy

You can install these dependencies using pip:

```pip install opencv-python numpy```


You will also need to download the YOLOv3 configuration file and weights from the official YOLO website:

- [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

Make sure that the configuration file and weights are located in the same directory as your Python script.

## Usage

Before running the script, you will need to start a video stream using `ffmpeg`. You can use the following command to start a video stream from a webcam and stream it to a UDP address:

``ffmpeg -f v4l2 -i /dev/video0 -preset ultrafast -tune zerolatency -f mpegts udp://192.168.5.185:1234``


This command starts a video stream from `/dev/video0` (your webcam) and streams it to `udp://192.168.5.185:1234`.

To use this script, you will need to modify the `url` variable in the script to match the URL of the video stream you want to use. In this case, the URL should be:

``url = 'udp://192.168.5.185:1234'``


You can also modify the `conf_threshold` and `iou_threshold` variables to adjust the detection sensitivity and non-maximum suppression overlap threshold, respectively.

Once you have modified the variables, you can run the script using the following command:

``python main.py``


This will open the video stream and start detecting objects in real-time. Detected objects will be highlighted with bounding boxes and labeled with their class and confidence score. To exit the script, press the 'q' key.

## Acknowledgements

This script uses the pre-trained YOLOv3 model, which was developed by the Darknet team. The original YOLOv3 paper can be found [here](https://arxiv.org/abs/1804.02767).
