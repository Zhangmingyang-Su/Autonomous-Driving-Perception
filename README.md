# Self-Driving Cars - Semantic Segmentation and Object Detection
This repository is for Computer Vision course project at George Washington University, The main purpose of this project is trying to complete Autonomous Driving Perception tasks to ensure safe driving condition.

## Team Member
* Zhangmingyang Su, Ze Gong, Derasari Preet


## Package Installation
* Tensorflow
* OpenCV
* yolov3.cfg
* coco.names
* yolov3.weights([Link](https://www.kaggle.com/valentynsichkar/yolo-coco-data?select=yolov3.cfg) because of large file)

## Semantic Segmentation
### DeepLab Model Architecture
![](pic/DeepLab%20Architecture.png)

### Segmentation Result
After Implementing DeepLab model, The driving scene is segmented into 16 categories which represented by different colors in real-time.  
![](pic/mit-driveSeg.gif) 
![](pic/segmentation_result.gif)

### Segmentation result based on different frame rates 
![](pic/different%20frame%20rate%20analysis.png) 

From the graph, it's easy to figure out the higher frame rate is, the better performance is. There are lots of things happen in a short-time, in order to ensure safe driving condition, we need to capture more information from frame to frame.  

### Segmentation result of original data vs temporal Data
![](pic/original%20vs%20temporal.png)

For the temporal data, such as the transformation of the color, background, distortion, etc. All of these frames are not independent, so, we sum the current frame result and previous frame result to smooth the prediction. By using the temporal data, the performance is a little bit better than the original one without the temporal data.

## Object Detection
### YOLO-V3
![](pic/yolov3.png)

### Object Detection Result
![](pic/frame%201.png)
![](pic/frame%202.png)
