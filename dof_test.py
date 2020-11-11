import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def calcDenseOpticalF(frame1, frame2):
    """
    Calculate dense optical flow from frame1 to frame 2
    :param frame1: start frame for DOF
    :param frame2: end frame for DOF
    :return: Optical flow
    """
    frame_prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(frame_prvs, frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def getRGB_DOF(flow, frame1):
    """
    Convert complex optical flow to RGB
    :param flow: flow to convert
    :param frame1: frame with right shape for RGB dof
    :return: RGB dof
    """
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[...,2] = mag*3
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb


def get_ydata(pathy):
    """
    Read labels from a file
    :param pathy: Path to label file
    :return: array containing labels
    """
    ydata_c = []
    with open(pathy, "r") as f:
        for line in f:
                ydata_c.append([float(line.strip("\n"))])
    ydata_c = np.array(ydata_c)
    return ydata_c

def get_generator(pathx, ydata, no_of_frames, start_frame):
    """
    Generator to get the frames from memory and calculate optical flow
    :param pathx: Path to video data
    :param ydata: Array containing labels
    :param no_of_frames: Number of frames to get in total
    :param start_frame: Specify start frame number
    """
    cap = cv2.VideoCapture(pathx)
    ret, frame0 = cap.read()
    frame0 = frame0[200:350]
    ret, frame1 = cap.read()
    frame1 = frame1[200:350]
    frame_num = 0
    X, y = [], []
    while frame_num <= no_of_frames+start_frame:
        frame_num += 1
        if frame_num > start_frame:
            if (frame_num - start_frame) % 100 == 0:
                print("antoher one")
            flow = calcDenseOpticalF(frame0, frame1)
            flow = getRGB_DOF(flow, frame1)
            y = ydata[frame_num][0]
            X = np.array(flow)
            yield X, y
        frame0 = frame1
        ret, frame1 = cap.read()
        frame1 = frame1[200:350]
    cap.release()

pathx = "data//test.mp4"
pathy = "data//train.txt"

y_dat = get_ydata(pathy)
gen = get_generator(pathx, y_dat, no_of_frames=10780, start_frame=0)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("dof_eval.avi", fourcc, 20, (640,150))
#csv = open("test.csv", "w")
#csv.write(f"speed;\n")

for fr, label in gen:
    out.write(fr.astype("uint8"))
 #   csv.write(f"{label}\n")
out.release()

#gen = get_generator(pathx, y_dat, no_of_frames=2000, start_frame=10000)
#for fr, label in gen:
#    out.write(fr.astype("uint8"))
 #   csv.write(f"{label}\n")

#gen = get_generator(pathx, y_dat, no_of_frames=2000, start_frame=18000)
#for fr, label in gen:
#    out.write(fr.astype("uint8"))
 #   csv.write(f"{label}\n")

#csv.close()



