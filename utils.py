import cv2
import numpy as np
import tensorflow as tf

def calcDenseOpticalF(frame1, frame2, show=0):
    frame_prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(frame_prvs, frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def getRGB_DOF(flow):
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return rgb


class SpeedDataset_2(tf.data.Dataset):
    def get_ydata(pathy):
        ydata_c = []
        with open(pathy, "r") as f:
            for line in f:
                ydata_c.append([float(line.strip("\n"))])
        ydata_c = np.array(ydata_c)
        return ydata_c

    def get_generator(pathx, ydata, no_of_frames, preprocess_func, start_frame):
        def frame_generator():
            cap = cv2.VideoCapture(pathx)
            ret, frame = cap.read()
            frame = frame[200:350]
            frame_num = 0
            X, y = [], []
            while frame_num < no_of_frames+start_frame:
                frame_num += 1
                if frame_num >= start_frame:
                    y.append(ydata[frame_num])
                    x = preprocess_func(frame)
                    X.append(x)
                    if len(X) == 10:
                        yield (np.array(X), np.array(y))
                        X = X[1:]
                        y = y[1:]
                ret, frame = cap.read()
                frame = frame[200:350]
            cap.release()
        return frame_generator

    def __new__(cls, pathx, pathy, no_of_frames=1000, preprocess_func=lambda x: x, start_frame=0):
        ydata = cls.get_ydata(pathy)
        out_shape = ((10, 150, 640, 3), (10, 1))
        out_types=(tf.float64, tf.float64)
        gen = cls.get_generator(pathx, ydata, no_of_frames, preprocess_func, start_frame)
        return tf.data.Dataset.from_generator(gen,
                                              output_types=out_types,
                                              output_shapes=out_shape)

class DoFDataset(tf.data.Dataset):
    def get_ydata(pathy):
        ydata_c = []
        with open(pathy, "r") as f:
            f.readline()
            for line in f:
                ydata_c.append([float(line.strip(";\n"))])
        ydata_c = np.array(ydata_c)
        return ydata_c

    def get_generator(pathx, ydata, preprocess_func):
        def frame_generator():
            cap = cv2.VideoCapture(pathx)
            ret, frame = cap.read()
            frame_num = 0
            while ret:
                y = ydata[frame_num]
                X = np.array(frame)
                X = preprocess_func(X)
                yield (X, y)
                frame_num += 1
                ret, frame = cap.read()
            cap.release()
        return frame_generator

    def __new__(cls, pathx, pathy, preprocess_func=lambda x: x):
        ydata = cls.get_ydata(pathy)
        out_shape = ((150, 640, 3), (1,))
        out_types=(tf.float64, tf.float64)
        gen = cls.get_generator(pathx, ydata, preprocess_func)
        return tf.data.Dataset.from_generator(gen,
                                              output_types=out_types,
                                              output_shapes=out_shape)


class DoFDataset_test(tf.data.Dataset):
    def get_generator(pathx, preprocess_func):
        def frame_generator():
            cap = cv2.VideoCapture(pathx)
            ret, frame = cap.read()
            frame_num = 0
            while ret:
                X = np.array(frame)
                X = preprocess_func(X)
                yield X
                frame_num += 1
                ret, frame = cap.read()
            cap.release()
        return frame_generator

    def __new__(cls, pathx, preprocess_func=lambda x: x):
        out_shape = (150, 640, 3)
        out_types=(tf.float64)
        gen = cls.get_generator(pathx, preprocess_func)
        return tf.data.Dataset.from_generator(gen,
                                              output_types=out_types,
                                              output_shapes=out_shape)

class SpeedDataset(tf.data.Dataset):
    def get_ydata(pathy):
        ydata_c = []
        with open(pathy, "r") as f:
            for line in f:
                ydata_c.append([float(line.strip("\n"))])
        ydata_c = np.array(ydata_c)
        return ydata_c

    def get_generator(pathx, ydata, no_of_frames, preprocess_func, start_frame):
        def frame_generator():
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
                    flow = calcDenseOpticalF(frame0, frame1)
                    y = ydata[frame_num]
                    X = np.array(flow)
                    X = preprocess_func(X)
                    yield (X, y)
                frame0 = frame1
                ret, frame1 = cap.read()
                frame1 = frame1[200:350]
            cap.release()
        return frame_generator

    def __new__(cls, pathx, pathy, no_of_frames=1000, preprocess_func=lambda x: x, start_frame=0):
        ydata = cls.get_ydata(pathy)
        out_shape = ((150, 640, 2), (1,))
        out_types=(tf.float64, tf.float64)
        gen = cls.get_generator(pathx, ydata, no_of_frames, preprocess_func, start_frame)
        return tf.data.Dataset.from_generator(gen,
                                              output_types=out_types,
                                              output_shapes=out_shape)

class SpeedDataset_test(tf.data.Dataset):
    def get_generator(pathx, no_of_frames, preprocess_func, start_frame):
        def frame_generator():
            cap = cv2.VideoCapture(pathx)
            ret, frame0 = cap.read()
            frame0 = frame0[200:350]
            ret, frame1 = cap.read()
            frame1 = frame1[200:350]
            frame_num = 0
            X = []
            while frame_num <= no_of_frames+start_frame:
                frame_num += 1
                if frame_num > start_frame:
                    flow = calcDenseOpticalF(frame0, frame1)
                    X = np.array(flow)
                    X = preprocess_func(X)
                    yield X
                frame0 = frame1
                ret, frame1 = cap.read()
                frame1 = frame1[200:350]
            cap.release()
        return frame_generator

    def __new__(cls, pathx, no_of_frames=1000, preprocess_func=lambda x: x, start_frame=0):
        print(pathx)
        out_shape = (150, 640, 2)
        out_types=(tf.float64)
        gen = cls.get_generator(pathx, no_of_frames, preprocess_func, start_frame)
        return tf.data.Dataset.from_generator(gen,
                                              output_types=out_types,
                                              output_shapes=out_shape)
