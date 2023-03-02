# Human Detection using YOLOv3

This script uses the YOLOv3 object detection model to detect humans in a video stream. It can be used with any video stream, as long as the URL of the stream is specified in the script.

## Requirements

To run this script, you will need to have the following dependencies installed:

- OpenCV
- NumPy

You can install these dependencies using pip:

`pip install opencv-python numpy`


## Usage

To use this script, you will need to modify the `url` variable in the script to match the URL of the video stream you want to use.

Once you have modified the `url` variable, you can run the script using the following command:

`python human_detection.py`


This will open the video stream and start detecting humans in real-time. To exit the script, press the 'q' key.

## Acknowledgements

This script uses the pre-trained YOLOv3 model, which was developed by the Darknet team. The original YOLOv3 paper can be found [here](https://arxiv.org/abs/1804.02767). You can download the YOLOv3 weights from [here](https://pjreddie.com/media/files/yolov3.weights).


