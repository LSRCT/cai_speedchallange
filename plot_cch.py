import numpy as np
import os
import tensorflow as tf
import preprocessing as pp
import utils
import cv2


def save_infer(gen, model, n_o_f=4000, save_path="vid_infer.csv"):
    fr = 0
    print(f"Running inference on video, saving to {save_path}")
    with open(save_path, "w") as f:
        header = f"frame;pred;\n"
        f.write(header)
        for sample in gen:
            pred = model.predict(sample)[0][0]
            f.write(f"{fr};{pred};\n")
            fr += 1
            if fr > n_o_f:
                break
            if fr%100 == 0:
                print(f"Done with {fr}/{n_o_f} frames")
    print("Done with video")

def annotate_video(p_pred, vid_path, s_path, n_o_f):
    print(p_pred[0])
    print(np.shape(p_pred))
    p_p = [f"Pred: {float(x)*1.609} km/h" for x in p_pred]

    cv2.CAP_PROP_CONVERT_RGB = False
    cap = cv2.VideoCapture(vid_path)
    save_name = f"{s_path}//" + "an_"+vid_path.split("/")[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_name, fourcc, 20.0,(640,480))

    ret, frame = cap.read()
    pred_list = []
    fnum = 0
    # for every frame
    while ret:
        pic = frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pic, p_p[fnum],(150,20), font, 0.5,(255,255,255), 1,cv2.LINE_AA)
        out.write(pic)
        fnum += 1
        if fnum == n_o_f:
            break
        ret, frame = cap.read()
    cap.release()        


def loadPred(path):
    pred = []
    with open(path, "r") as f:
        # first one is header
        f.readline()
        # other ones are data
        for line in f:
            ls = line.strip("\n")
            ls = ls.split(";")[1:]
            pred.append(float(ls[0]))
    return np.array(pred)


def get_normalized_data(pathx, no_of_frames):
    # normalize data
    dataset = utils.SpeedDataset_test(pathx_test, no_of_frames=no_of_frames)
    dataset = dataset.batch(10, drop_remainder=True)
    dat_avg = pp.calcAVG_generator_test(dataset, verbose=1)

    dataset = utils.SpeedDataset_test(p3thx_test, no_of_frames=no_of_frames)
    dataset = dataset.batch(10, drop_remainder=True)
    data_std = pp.calcSTDDEV_generator_test(dataset, dat_avg, verbose=1)
    
    dataset = utils.SpeedDataset_test(pathx_test, no_of_frames=no_of_frames, start_frame=0, preprocess_func=pp.preprocess(data_std, dat_avg))
    dataset = dataset.batch(1, drop_remainder=True)
    return dataset

if __name__ == "__main__":
    pathx = "data3//dof_train.avi"
    pathy = "data3//trainY.csv"
    pathx_test = "data//test.mp4"
    no_of_frames = 9999

    savedir = os.getcwd()+"//models//resnet_100_3"
    model = tf.keras.models.load_model(savedir+"//clust_model.h5", compile=True)
    #model.compile(optimizer="adam",
   #               loss='MSE')

    model.summary()

    pathx_t = "data3//dof_train.avi"
    mini_batch_size = 1

    ## normalize data
    dataset = utils.DoFDataset_test(pathx_t)
    dataset = dataset.batch(5, drop_remainder=True)
    dat_avg = pp.calcAVG_generator_test(dataset, verbose=1)

    dataset = utils.DoFDataset_test(pathx_t)
    dataset = dataset.batch(5, drop_remainder=True)
    data_std = pp.calcSTDDEV_generator_test(dataset, dat_avg, verbose=1)

    pathx_t = "data3//dof_eval.avi"
    dataset = utils.DoFDataset_test(pathx_t, preprocess_func=pp.preprocess(data_std, dat_avg))

    dataset = dataset.batch(mini_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(3)
    dataset = dataset.repeat()
    save_infer(dataset, model, save_path=savedir+"//vid_infer.csv", n_o_f = no_of_frames)
    pred = loadPred(savedir+"//vid_infer.csv")
    # moving average
    m_width = 20
    pred = np.convolve(pred, np.ones((m_width,), dtype=np.float64), mode="same")/m_width
    annotate_video(pred, pathx_test, savedir, no_of_frames)

