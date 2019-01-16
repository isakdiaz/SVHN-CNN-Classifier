import numpy as np

import h5py
import os
import cv2


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def create_pyramid(img, layers=3):
    print("Original Image Size", img.shape)
    pyramid = []
    for i in range(layers):
        resize =  1 / (2 ** i)
        temp_image = np.copy(cv2.resize(img, (0, 0), fx=resize, fy=resize))
        pyramid.append(temp_image)
        print("Image size", img.shape, temp_image.shape)
        # cv2.imshow('Color image', temp_image / 255.)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return np.array(pyramid)


def window_cutouts(pyramid_imgs, size = 48, stride = 12):

    print("Pyramid Shape", pyramid_imgs.shape)
    #R is the inverse pyramid ratio
    #This function creates a multilayer pyramid and returns bounding boxes with
    #coordinates for the original size image

    nn_size = 48  #Size that neural network will use as input
    wndw_imgs = []
    wndw_loc = []


    # Find cutouts of each possible image to test
    for x, bgr_img in enumerate(pyramid_imgs):
        stride -= 2 # Reduce Stride in smaller images
        R = (2 ** x)
        print("IMAGE SHAPE", bgr_img.shape, " R: ", R)
        h, w, channels = bgr_img.shape

        for i in range(0, h - size, stride):
            for j in range(0, w - size, stride):

                temp_img = cv2.resize((bgr_img[i:i + size, j:j + size]), (nn_size, nn_size))
                wndw_imgs.append(temp_img)
                wndw_loc.append([R*i, R*j, R*size])



    return np.array(wndw_imgs), np.array(wndw_loc)

if __name__ == "__main__":

    numpy_save = True
    img_file = 'samples/report.png'

    #Read Image, Turn to grayscale and subtract mean
    image = np.array(cv2.imread(img_file))
    temp_img = np.copy(image) - np.mean(image)
    pyramid = create_pyramid(temp_img)
    wndw_imgs, wndw_loc = window_cutouts(pyramid)


    print("WINDOW IMAGE SHAPE", wndw_imgs.shape)
    print("WINDOW LOC SHAPE", wndw_loc.shape)
    # for img in pyramid:
    #     cv2.imshow('Color image', img / 255.)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    if numpy_save:
        np.save("wndw_imgs.npy", wndw_imgs)
        np.save("wndw_loc.npy", wndw_loc)
