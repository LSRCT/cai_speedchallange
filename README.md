# Comma.ai speed challenge

I tried the speedchallenge by comma.ai (https://github.com/commaai/speedchallenge) for fun and profit.

The challenge here is to predict the speed of a driving car given dashcam footage.
They include a labeld train set containing ~20000 frames of data, as well as a test set containing 10000 frames.

## Basic approach

I choose to use dense optical flow with a small resnet as the basic approach. The alternative is see to that is a solution involving recurrent neural nets, but as the dataset is small and RNNs are typically harder to train I decided against that.

## Pipeline
the pipeline is pretty standard, containing the following steps

1. Preprocessing
2. Classification
3. Postprocessing

### Preprocessing
In the preprocessing step the frames are cropped, removing the sky and parts of the car (as seen in pic ...)
After that dense optical flow to the next frame is calculated.
Lastly the data is normalized to zero mean and unit variance.

### Classification 

The model used contains 3 CNN layers with residual connections as a feature extraction part and a two fully connected dense layers for the classification part.
![architecture](/media/architecture.jpg)

The parameters for the number of neurons in the classification layer as well as stride of the conv layers are on the low side, I choose them so the models doesnt get huge.

### Post processing
This is only done for inference, but there i chose to use a running mean with unit window and 20 frame width to smooth out the predctions a bit

## Training 

The validation split is a bit tricky since just splitting the video into 3/4 train and 1/4 validation will cause the validation set to not contain any highway frames, but only slower city frames.
Because of that, I choose to use the first 2000, another 2000 in the middle, as well as the last 2000 frames for validation, the frames in between for training. This provides a representative test set without being very correlated in time.

## Results 
The model was trained for 100 epochs, progression of train and validation MSE can be seen in the graph below. The final validation loss is ~6.9 mph MSE.
![loss](/media/epoch_loss.jpg)

## Implementation
Implementation is done in Python3 using:
- Tensorflow, Keras
- OpenCV 2
- numpy, scipy, matplotlib
