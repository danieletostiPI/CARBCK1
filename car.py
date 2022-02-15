# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import time
import RPi.GPIO as GP

GP.setmode(GP.BOARD)
GP.setwarnings(False)

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
# IM_WIDTH = 640
# IM_HEIGHT = 480

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

### Picamera ###
# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH, IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
rawCapture.truncate(0)
# -------------------------------------------------------------------------------------
# RPGIO:

LEDG = 16
BUTTON_STOP = 18
TRIGR = 7  # 38  # fa muovore a destra
ECHOR = 11  # 40

GP.setup(LEDG, GP.OUT)
GP.setup(ECHOR, GP.IN)
GP.setup(TRIGR, GP.OUT)

GP.setup(BUTTON_STOP, GP.IN, pull_up_down=GP.PUD_UP)  # Se Button = 0, bottone pigiato.

# MOTOR:    #HP: DIR1:HIGH e DIR2:LOW il motore gira portando il veicolo in avanti

RR, RR_DIR1, RR_DIR2 = 40, 38, 36  # REAR RIGHT
RL, RL_DIR1, RL_DIR2 = 33, 35, 37

FR, FR_DIR1, FR_DIR2 = 29, 23, 21
FL, FL_DIR1, FL_DIR2 = 26, 24, 22

# MOTOR:    #HP: DIR1:HIGH e DIR2:LOW il motore gira portando il veicolo in avanti
GP.setup(RR, GP.OUT)
GP.setup(RR_DIR1, GP.OUT)
GP.setup(RR_DIR2, GP.OUT)

GP.setup(RL, GP.OUT)
GP.setup(RL_DIR1, GP.OUT)
GP.setup(RL_DIR2, GP.OUT)

GP.setup(FR, GP.OUT)
GP.setup(FR_DIR1, GP.OUT)
GP.setup(FR_DIR2, GP.OUT)

GP.setup(FL, GP.OUT)
GP.setup(FL_DIR1, GP.OUT)
GP.setup(FL_DIR2, GP.OUT)

PWM_RR = GP.PWM(RR, 100)  # set pwm for each motor
PWM_RL = GP.PWM(RL, 100)
PWM_FR = GP.PWM(FR, 100)
PWM_FL = GP.PWM(FL, 100)


def go_bw(pwm_default):
    GP.output(RR_DIR1, GP.HIGH)
    GP.output(RR_DIR2, GP.LOW)
    PWM_RR.ChangeDutyCycle(pwm_default)

    GP.output(RL_DIR1, GP.HIGH)
    GP.output(RL_DIR2, GP.LOW)
    PWM_RL.ChangeDutyCycle(pwm_default)

    GP.output(FR_DIR1, GP.LOW)
    GP.output(FR_DIR2, GP.HIGH)
    PWM_FR.ChangeDutyCycle(pwm_default)

    GP.output(FL_DIR1, GP.HIGH)
    GP.output(FL_DIR2, GP.LOW)
    PWM_FL.ChangeDutyCycle(pwm_default)


def go_fw(pwm_default):  # velocità predefinita

    GP.output(RR_DIR1, GP.LOW)
    GP.output(RR_DIR2, GP.HIGH)
    PWM_RR.ChangeDutyCycle(pwm_default)

    GP.output(RL_DIR1, GP.LOW)
    GP.output(RL_DIR2, GP.HIGH)
    PWM_RL.ChangeDutyCycle(pwm_default)

    GP.output(FR_DIR1, GP.HIGH)
    GP.output(FR_DIR2, GP.LOW)
    PWM_FR.ChangeDutyCycle(pwm_default)

    GP.output(FL_DIR1, GP.LOW)
    GP.output(FL_DIR2, GP.HIGH)
    PWM_FL.ChangeDutyCycle(pwm_default)


def go_right(pwm_default):
    GP.output(RR_DIR1, GP.HIGH)
    GP.output(RR_DIR2, GP.LOW)
    PWM_RR.ChangeDutyCycle(pwm_default)

    GP.output(RL_DIR1, GP.LOW)
    GP.output(RL_DIR2, GP.HIGH)
    PWM_RL.ChangeDutyCycle(pwm_default)

    GP.output(FR_DIR1, GP.LOW)
    GP.output(FR_DIR2, GP.HIGH)
    PWM_FR.ChangeDutyCycle(pwm_default)

    GP.output(FL_DIR1, GP.LOW)
    GP.output(FL_DIR2, GP.HIGH)
    PWM_FL.ChangeDutyCycle(pwm_default)


def go_left(pwm_default):
    GP.output(RR_DIR1, GP.LOW)
    GP.output(RR_DIR2, GP.HIGH)
    PWM_RR.ChangeDutyCycle(pwm_default)

    GP.output(RL_DIR1, GP.HIGH)
    GP.output(RL_DIR2, GP.LOW)
    PWM_RL.ChangeDutyCycle(pwm_default)

    GP.output(FR_DIR1, GP.HIGH)
    GP.output(FR_DIR2, GP.LOW)
    PWM_FR.ChangeDutyCycle(pwm_default)

    GP.output(FL_DIR1, GP.HIGH)
    GP.output(FL_DIR2, GP.LOW)
    PWM_FL.ChangeDutyCycle(pwm_default)


def turn_left():
    GP.output(RR_DIR1, GP.LOW)
    GP.output(RR_DIR2, GP.HIGH)
    PWM_RR.ChangeDutyCycle(100)

    GP.output(RL_DIR1, GP.LOW)
    GP.output(RL_DIR2, GP.HIGH)
    PWM_RL.ChangeDutyCycle(50)

    GP.output(FR_DIR1, GP.HIGH)
    GP.output(FR_DIR2, GP.LOW)
    PWM_FR.ChangeDutyCycle(100)

    GP.output(FL_DIR1, GP.LOW)
    GP.output(FL_DIR2, GP.HIGH)
    PWM_FL.ChangeDutyCycle(50)


def turn_right():
    GP.output(RR_DIR1, GP.LOW)
    GP.output(RR_DIR2, GP.HIGH)
    PWM_RR.ChangeDutyCycle(50)

    GP.output(RL_DIR1, GP.LOW)
    GP.output(RL_DIR2, GP.HIGH)
    PWM_RL.ChangeDutyCycle(100)

    GP.output(FR_DIR1, GP.HIGH)
    GP.output(FR_DIR2, GP.LOW)
    PWM_FR.ChangeDutyCycle(50)

    GP.output(FL_DIR1, GP.LOW)
    GP.output(FL_DIR2, GP.HIGH)
    PWM_FL.ChangeDutyCycle(100)


spin = 0
turn = 0
stop = True
i = 0
dnew = 0
dold = 0
pwm_default = 95  # %
pwm_go = 95
pwm_back = 95
area = 0
area_old = 0
timee = 0
time_old = 0
TOT = 0
xold = 0

PWM_RR.start(0)  # set initial value of pwms
PWM_RL.start(0)
PWM_FR.start(0)
PWM_FL.start(0)

PWM_RR.ChangeDutyCycle(0)
PWM_RL.ChangeDutyCycle(0)
PWM_FR.ChangeDutyCycle(0)
PWM_FL.ChangeDutyCycle(0)

# -------------------------------------------------------------------------------------
for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    t1 = cv2.getTickCount()

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.copy(frame1.array)
    frame.setflags(write=1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    # frame,
    # np.squeeze(boxes),
    # np.squeeze(classes).astype(np.int32),
    # np.squeeze(scores),
    # category_index,
    # use_normalized_coordinates=True,
    # line_thickness=8,
    # min_score_thresh=0.40)

    if int(classes[0][0]) == 1 and scores[0][0] > 0.4:
        xcenter = int(((boxes[0][0][1] + boxes[0][0][3]) / 2) * IM_WIDTH)
        ycenter = int(((boxes[0][0][0] + boxes[0][0][2]) / 2) * IM_HEIGHT)
        area = ((boxes[0][0][3] - boxes[0][0][1]) * IM_WIDTH) * ((boxes[0][0][2] - boxes[0][0][0]) * IM_HEIGHT)

    elif (int(classes[0][1]) == 1 and scores[0][1] > 0.4):
        xcenter = int(((boxes[0][1][1] + boxes[0][1][3]) / 2) * IM_WIDTH)
        ycenter = int(((boxes[0][1][0] + boxes[0][1][2]) / 2) * IM_HEIGHT)
        area = ((boxes[0][1][3] - boxes[0][1][1]) * IM_WIDTH) * ((boxes[0][1][2] - boxes[0][1][0]) * IM_HEIGHT)

    elif (int(classes[0][2]) == 1 and scores[0][2] > 0.4):
        xcenter = int(((boxes[0][2][1] + boxes[0][2][3]) / 2) * IM_WIDTH)
        ycenter = int(((boxes[0][2][0] + boxes[0][2][2]) / 2) * IM_HEIGHT)
        area = ((boxes[0][2][3] - boxes[0][2][1]) * IM_WIDTH) * ((boxes[0][2][2] - boxes[0][2][0]) * IM_HEIGHT)

    else:
        xcenter = 0
        ycenter = 0
        area = 0

    print("X = ", xcenter)
    print("Y = ", ycenter)
    ## puoi creare una funzione che se per piu di tot frames non rileva una persona allora vuol dire che la persona non c'è
    ## e devo iniziare a cercarla, altrimenti tengo la xcenter, ycenter del giro prima

    # All the results have been drawn on the frame, so it's time to display it.
    # cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    timetot = (t2 - t1) / freq
    frame_rate_calc = 1 / timetot
    print(frame_rate_calc)
    key = cv2.waitKey(1) & 0xFF


    ## Action:

    # Distance :
    # dnew = calculate_distance_r()
    # time.sleep(0.005)

    # if Se sono fermo e pwm current = 0 localizza in posizione centrale schermo e avvia motor:--------------

    if xcenter > IM_WIDTH / 2 - IM_WIDTH / 7 and xcenter < IM_WIDTH / 2 + IM_WIDTH / 7:
        GP.output(LEDG, GP.HIGH)
        timee = 0
        if (IM_WIDTH * IM_HEIGHT) / 4 > area > 0:
            pwm_default = pwm_go  # %
            go_fw(pwm_default)
        elif area > (IM_WIDTH * IM_HEIGHT) / 3:
            pwm_default = pwm_back  # %
            go_bw(pwm_default)
        else:
            pwm_default = 0  # %
            go_fw(pwm_default)

        area_old = area

    elif xcenter > IM_WIDTH / 2 + IM_WIDTH / 7:
        GP.output(LEDG, GP.HIGH)
        timee = 0
        if pwm_default == 0 and 0 < area < IM_WIDTH * IM_HEIGHT / 2:
            print("gira a destra")
            go_right(pwm_go)
            time.sleep(((xcenter - IM_WIDTH / 2) / (IM_WIDTH / 2)) * 0.5)
            go_fw(pwm_default)
        elif pwm_default == pwm_go:
            print("Gira a destra in movimento")
            turn_right()

    elif xcenter < IM_WIDTH / 2 - IM_WIDTH / 7 and xcenter > 0:
        GP.output(LEDG, GP.HIGH)
        timee = 0
        if pwm_default == 0 and 0 < area < IM_WIDTH * IM_HEIGHT / 2:
            print("gira a sinistra")
            # dc1 = rotate_right(dc1, inc)
            go_left(pwm_go)
            time.sleep(((IM_WIDTH / 2 - xcenter) / (IM_WIDTH / 2)) * 0.5)
            go_fw(pwm_default)
        elif pwm_default == pwm_go:
            print("Gira a sinistra in movimento")
            turn_left()

    elif xcenter == 0 and ycenter == 0:
        if timee == 0:
            GP.output(LEDG, GP.LOW)
            pwm_default = 0  # %
            go_fw(pwm_default)
            time1 = 0
            time1 = cv2.getTickCount()
            timee = 1
            i = 0

        time2 = cv2.getTickCount()
        tottime = (time2 - time1) / freq
        TOT = tottime - time_old
        i = i + 1
        if TOT > 1.5 and i > 3:
            print("TOT TIME: ", TOT)
            time_old = tottime
            tottime = 0
            TOT = 0
            timee = 0
            time2 = 0
            if xold < IM_WIDTH / 2:  # ruota nella direzione dell'ultima posizione utile nota della persona in questione
                go_left(pwm_go)
            elif xold > IM_WIDTH / 2:
                go_right(pwm_go)
            time.sleep(0.6)
            go_fw(0)

    if xcenter != 0:
        xold = xcenter

    # -------------------------------------------------------------------------------------------------------

    # elif se sono in movimento e pwm current != 0 gira poco in movimento:

    rawCapture.truncate(0)

    # Press 'q' to quit         ########## prova a togliere questo comando per vedere se funziona
    # if key == ord('q'):
    # break

    # --------------------------
    stop = GP.input(BUTTON_STOP)
    if not stop:
        GP.output(LEDG, GP.HIGH)
        time.sleep(0.5)
        GP.output(LEDG, GP.LOW)
        time.sleep(0.5)
        GP.output(LEDG, GP.HIGH)
        print("Well Done")
        break
    # --------------------------

camera.close()
