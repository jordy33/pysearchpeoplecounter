# People Counter
### Detach raspberry pi camera
### Install Jetson Nano Developer Kitl SDK4.1
```
https://developer.nvidia.com/embedded/jetpack
```
Configure using :

User: jetson Password: jetson

### Test CSI Camera
```
https://github.com/JetsonHacksNano/CSI-Camera
```
### install the basics

```
$ mkdir -p ${HOME}/project
$ cd ${HOME}/project
$ git clone https://github.com/jkjung-avt/jetson_nano.git
$ cd jetson_nano/
$ ./install_basics.sh
```
### install 4Gb swapfile
```
mkdir ~/project
cd ~/project
git clone https://github.com/JetsonHacksNano/installSwapfile 
cd installSwapfile
edit installSwapfile.sh --> change swapfile from 6gbto 4g
./installSwapfile.sh
```

## Install Environment

### install opencv 3.4.0
```
sudo nvpmodel -m 0
sudo jetson_clocks
cd ${HOME}/project/jetson_nano
./install_opencv-3.4.6.sh
```

### Install Dlib
```
sudo pip3 install dlib
# Install imutils
sudo pip3 install imutils
```

```
git clone https://github.com/jordy33/peoplecounter.git
```
### Install scipy
```
sudo pip3 install scipy

```
#GPU graph
```
git clone https://github.com/jetsonhacks/gpuGraphTX
sudo apt-get install python-matplotlib
./gpuGraph.py
```

### Install jetcam
```
git clone https://github.com/NVIDIA-AI-IOT/jetcam
cd jetcam
sudo python3 setup.py install
```

### Execute People counter

```
$ cd ${HOME}
https://github.com/jordy33/peoplecounter.git
cd peoplecounter
```

### Execute the code 

```
python3 people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=21/1, format=NV12 ! nvvidconv flip-method=2 ! video/x-raw, width=960, height=616 format=BGRx ! videoconvert ! appsink'
```

### Install Jupyter Lab
```
sudo apt install nodejs npm
sudo pip3 install jupyter jupyterlab
sudo jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab --generate-config
```

### To test run
```
jupyter notebook --ip='0.0.0.0' --no-browser --log-level=0 --notebook-dir=/home/jetson
```

### Create File
```
sudo vim /etc/systemd/system/jupyter.service
```
### And insert:
```
[Unit]
Description=Jupyter Workplace

[Service]
Type=simple
PIDFile=/run/jupyter.pid
ExecStart=/usr/local/bin/jupyter notebook --ip='0.0.0.0' --no-browser --log-level=0 --notebook-dir=/home/jetson
User=jetson
Group=jetson
WorkingDirectory=/home/jetson
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Install
```
sudo systemctl enable jupyter.service
sudo systemctl daemon-reload
sudo systemctl restart jupyter.service
```

### Open Jupiter lab

```
http://<ip>:8888
```

### Create a Notebook and insert the following code:

```
from jetcam.csi_camera import CSICamera
import ipywidgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg

#
camera = CSICamera(width=224, height=224)
image = camera.read()
#
image_widget = ipywidgets.Image(format='jpeg')

image_widget.value = bgr8_to_jpeg(image)

display(image_widget)

camera.running = True

def update_image(change):
    image = change['new']
    image_widget.value = bgr8_to_jpeg(image)
    
camera.observe(update_image, names='value')

#To stop it, unattach the callback with the unobserve method.

camera.unobserve(update_image, names='value')


#Another way to view the image stream
import traitlets

camera_link = traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)

#You can remove the camera/widget link with the unlink method.

camera_link.unlink()

#... and reconnect it again with link.

camera_link.link()


#Shut down the kernel of this notebook to release the camera resource.

#To do so, shut down the notebook's kernel from the JupyterLab pull-down menu: Kernel->Shutdown Kernel, then restart it with #Kernel->Restart Kernel.


#If the camera setup appears "stuck" or the images "frozen", follow these steps:

#Shut down the notebook kernel as explained above
#Open a terminal on the Jetson Nano by clicking the "Terminal" icon on the "Launch" page
#Enter the following command in the terminal window: sudo systemctl restart nvargus-daemon with password:dlinano
```
### part1
```
%pylab inline 
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from IPython.display import clear_output
```
### Part 2
```
args = {}
args["prototxt"]="mobilenet_ssd/MobileNetSSD_deploy.prototxt"
args["model"]="mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
args["input"]="'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=21/1, format=NV12 ! nvvidconv flip-method=2 ! video/x-raw, width=960, height=616 format=BGRx ! videoconvert ! appsink'"
args["output"]=""
args["confidence"]=0.4
args["skip_frames"]=30
```
### Part 3
```
# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Waiting"
	rects = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue

				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	axis('off')
	# Title of the window
	title("Input Stream")
	# Display the frame
	imshow(frame)
	show()
	# Display the frame until new frame is available
	clear_output(wait=True)
	#cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()
```
NEW PC
```
%pylab inline 
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
```
p2
```
from jetcam.csi_camera import CSICamera
import ipywidgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg
```
p3
```
args = {}
args["prototxt"]="mobilenet_ssd/MobileNetSSD_deploy.prototxt"
args["model"]="mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
args["input"]="'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=21/1, format=NV12 ! nvvidconv flip-method=2 ! video/x-raw, width=960, height=616 format=BGRx ! videoconvert ! appsink'"
args["output"]=""
args["confidence"]=0.4
args["skip_frames"]=30
```
p4
```
camera = CSICamera(width=224, height=224)
image = camera.read()
```
p5
```
def update_image(change):
	image = change['new']
	info = [
		("Up", 0),
		("Down", 0),
		("Status", 0),
	]
	(H, W) = image.shape[:2]
	# if we are supposed to be writing a video to disk, initialize
	# the writer
	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(image, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(args["output"], fourcc, 30,(W, H), True)
	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(image)
	image_widget.value = bgr8_to_jpeg(image)
```
p6
```
# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()
```
p7
```
image_widget = ipywidgets.Image(format='jpeg')
image_widget.value = bgr8_to_jpeg(image)
display(image_widget)
camera.running = True
camera.observe(update_image, names='value')
```
