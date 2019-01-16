import processing
import detection
import classification
import numpy as np
import cv2

if __name__ == "__main__":

     #Run for image files

    img_filenames = ['1.png', '2.png', '3.png', '4.png', '5.png']

    CNNmodel = None
    for img_file in img_filenames:

        #TODO Step 1 Processing

        numpy_save = False
        model_file = "VGG_Transfer_model.h5"
        weights_file = None


        #Read Image, Turn to grayscale and subtract mean
        image = np.array(cv2.imread(img_file))
        temp_img = np.copy(image) - np.mean(image)
        pyramid = processing.create_pyramid(temp_img)
        wndw_imgs, wndw_loc =  processing.window_cutouts(pyramid)


        print("WINDOW IMAGE SHAPE", wndw_imgs.shape)
        print("WINDOW LOC SHAPE", wndw_loc.shape)

        if numpy_save:
            np.save("wndw_imgs.npy", wndw_imgs)
            np.save("wndw_loc.npy", wndw_loc)

        #TODO Step 2 DETECTION


        print("TEST IMG SIZES", wndw_imgs.shape)


        if CNNmodel is None:
            predictions, CNNmodel = detection.CNN_model(test_dataset=wndw_imgs, model_file=model_file)
        else:
            predictions = np.array(CNNmodel.predict(x=wndw_imgs, batch_size=128))

        print("PREDICTIONS Finished")

        if numpy_save:
            np.save("predictions.npy", predictions)


        # TODO Step 3 CLASSIFICATION

        final_image = classification.run(img_file=img_file, predictions=predictions, \
                                         wndw_loc=wndw_loc, model = CNNmodel)
