import numpy as np
import h5py
import cv2
import os
from keras.models import Model, load_model


# TENSORFLOW: USE CPU INSTEAD OF GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def CNN_model(test_dataset, model_file, weights_file = None):

    #Load Model & Weights
    model = load_model(model_file)
    if weights_file is not None:
        model.load_weights(weights_file)

    #Find predicted Values of X
    pred_array = model.predict(x=test_dataset, batch_size=1)

    return np.array(pred_array), model



if __name__ == "__main__":

    model_file = "VGG_Transfer_model.h5"
    img_file = 'samples/report.png'

    save_numpy = True

    # wndw_loc = np.load('wndw_loc.npy')
    wndw_imgs = np.load('wndw_imgs.npy')


    print("TEST IMG SIZES", wndw_imgs.shape)


    predictions = CNN_model(test_dataset=wndw_imgs, model_file=model_file)
    print("PREDICTIONS Finished")

    if save_numpy:
        np.save("predictions.npy", predictions)


