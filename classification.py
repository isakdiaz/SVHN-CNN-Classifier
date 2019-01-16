import numpy as np
import h5py
import os
import cv2

from scipy import stats

def run(img_file, predictions, wndw_loc, model):

    output_file = 'graded_images/' + img_file


    print('Output file:', output_file)
    image = np.array(cv2.imread(img_file))
    final_image = np.copy(image)

    max_pred = np.argmax(predictions, 2).T
    print("max_pred Shape", max_pred.shape)

    #Find all rows that do not contain all zeros
    x_ind, y_ind = np.where(max_pred != 10)
    poi_ind = np.unique(x_ind)
    print("X ARRAY", poi_ind)



    poi_nums = []
    poi_loc = []

    def list2num(list):
        #Remove 0 values
        list  = list[np.isin(list, 10, invert=True)]
        temp_num=0
        for j in range(len(list)):
            temp_num += list[j] * 10 ** (len(list) - j - 1)
        return temp_num

    # Find Points of interest and Window locations
    # Calculate the number from an arrray of 5 possible values
    min_accuracy = 5 * 0.85
    for i in poi_ind:
        if np.sum(np.max(predictions[:,i,:], axis=1)) > min_accuracy:

            poi_list = max_pred[i]
            temp_num = list2num(poi_list)

            poi_nums.append(temp_num)
            poi_loc.append(wndw_loc[i])

    #If not point of interests found, return original Image:
    if len(poi_loc) == 0:
        cv2.imwrite(output_file, final_image)
        return final_image


    def find_center(image, positions):
        h, w, _ = image.shape
        template = np.zeros((h, w))
        print("IMAGE SHAPE, ",h,w)
        for y,x,size in positions:
            template[y:y+size, x:x+size] += 10

        max_point = np.max(template)
        print("Max Point", max_point)

        y, x = np.mean(np.where(template == max_point), axis=1)
        print("Average Location of MAX point",x, y )

        return int(x), int(y)

    centerX, centerY = find_center(image, poi_loc)
    print("CENTERS", centerX, centerY)

    print("NUMBER OF HITS", x_ind.shape)
    print("ALL", poi_nums)


    #DRAW ON IMAGE
    # for num, loc in poi_nums, poi_loc:
    for i in range(len(poi_nums)):
        y, x, size = poi_loc[i]

        if size == 48:
            color = (0,255,255)
        elif size == 96:
            color = (255, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.rectangle(image,(x,y), (x+size,y+size), color,1)
        cv2.putText(image, str(poi_nums[i]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,255,255),1,cv2.LINE_AA)

    print(centerX, centerY)
    cv2.circle(image, (centerX, centerY), 5, (0, 0, 255), -1)
    # cv2.imwrite("report_marked.png", image)


    poi_loc = np.array(poi_loc)
    x = np.zeros((poi_loc.shape[0], 2))
    poi_loc = np.append(poi_loc, x, axis=1)
    poi_loc[:,3] = poi_loc[:,0] + poi_loc[:,2]
    poi_loc[:,4] = poi_loc[:,1] + poi_loc[:,2]



    poi_loc = poi_loc.astype(np.int32)

    #Find all boxes that encompass the center point for each possible size
    for box_size in np.sort(np.unique(poi_loc[:,2])):
        print("BOX SIZES", np.sort(np.unique(poi_loc[:,2])))
        poi_small_pyramid = poi_loc[poi_loc[:, 2] == box_size]
        min_bbox = np.array([1, 1]) * np.inf
        max_bbox = np.array([1, 1]) * -1 * np.inf
        for minY, minX, size, maxY, maxX in poi_small_pyramid:
            # print("TEST1", minY, minX, size, maxY, maxX)
            if minX <= centerX <= maxX and minY <= centerY <= maxY:
                # print("TEST2", minY, minX, size, maxY, maxX)
                min_bbox = np.min((min_bbox, [minY, minX]), axis=0)
                max_bbox = np.max((max_bbox, [maxY, maxX]), axis=0)

        # Break loop when a pyramid size produces a bounding box
        if np.sum(max_bbox) > 0:
            print("Found Box, breaking loop at", size)
            break
        else:
            print("DIDNT FIND Box, breaking loop at", size)


    print("BOUNDING BOX", min_bbox, max_bbox)


    #Expand the bbox by all rectangles which touch it according to smallest pyramind size

    complete = False
    while not complete:
        complete = True
        for minY, minX, size, maxY, maxX in poi_small_pyramid:
            # print("MIN MAX", minY, minX, size, maxY, maxX)
            # print("BB BOX", minY ,min_bbox[0] , maxY, minY , max_bbox[0] , maxY)
            if minY < min_bbox[0] < maxY or minY < max_bbox[0] < maxY or \
                    min_bbox[0]< minY < max_bbox[0] or min_bbox[0]< maxY < max_bbox[0]:
                # print("Pass 1", minY, minX, size, maxY, maxX)
                if minX < min_bbox[1] < maxX or minX < max_bbox[1] < maxX or \
                        min_bbox[1] <  minX < min_bbox[1] or max_bbox[1] < maxX < max_bbox[1]:

                    # print("Pass 2", minY, minX, size, maxY, maxX)
                    min_bbox = np.min((min_bbox, [minY, minX]), axis=0)
                    max_bbox = np.max((max_bbox, [maxY, maxX]), axis=0)
                    complete = False
                    # print("NOT COMPLETE")


    print("BOUNDING BOX", min_bbox, max_bbox)


    #No Bounding box found, return original file
    if np.sum(min_bbox) == np.inf or np.sum(max_bbox) == (np.inf * -1):
        cv2.imwrite(output_file, final_image)
        return final_image

    #Increase the scale of the bounding box by specific Percentage and reclasify
    start_box = []
    end_box = []
    scale_sizes = [-0.1, 0.0, 0.1]

    #Find binding boxes of different scales
    for scale in scale_sizes:
        y1, x1 = min_bbox
        y2, x2 = max_bbox
        x_scale = (x2 - x1) * scale
        y_scale = (y2 - y1) * scale

        y1, x1 = (min_bbox - y_scale).astype(np.int32)
        y2, x2 = (max_bbox + x_scale).astype(np.int32)

        start_box.append([y1, x1])
        end_box.append([y2, x2])

        #Draw the bounding box on the image for the ZERO scale
        if scale == 0.0:
            bb_y1, bb_x1 = y1, x1
            bb_y2, bb_x2 = y2, x2


    cv2.rectangle(image, (bb_x1,bb_y1), (bb_x2,bb_y2), (255,255,255),1)
    cv2.rectangle(final_image, (bb_x1,bb_y1), (bb_x2,bb_y2), (0, 255, 0),1)

    # cv2.imwrite("report_marked.png", image)


    img = np.array(cv2.imread(img_file), dtype=np.float32)
    img -= np.mean(img)

    nn_size = 48
    rotations = [0, 30, -30, 45, -45, 60, -60]

    rotate_resize_imgs = []
    for i in range(len(start_box)):
        y1, x1 = start_box[i]
        y2, x2, = end_box[i]

        resized_img = cv2.resize((img[y1:y2, x1:x2]), (nn_size, nn_size))

        for degree in rotations:
            M = cv2.getRotationMatrix2D((nn_size / 2, nn_size / 2), degree, 1)
            rotated_img = cv2.warpAffine(resized_img, M, (nn_size, nn_size))
            rotate_resize_imgs.append(rotated_img)



    rotate_resize_imgs = np.asarray(rotate_resize_imgs)


    test_dataset = rotate_resize_imgs

    # for img in test_dataset:
    #     cv2.imshow('Color image', img / 255.)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # test_dataset = temp_array.reshape(temp_array.shape + (1,))
    print("TEST DATASET SHAPE", test_dataset.shape)
    import detection
    bb_predictions = np.array(model.predict(x=test_dataset, batch_size=16))
    max_pred = np.argmax(bb_predictions, 2).T
    print("BEST PREDICTIONS", max_pred)



    #Find the max prediction length detected and then return the mode of all values of that length
    max_length = np.max(np.sum(np.isin(max_pred, 10, invert=True),axis=1))
    max_len_pred = np.array([])
    for pred_list in max_pred:
        temp_num = list2num(pred_list)
        length = len(str(temp_num))
        if length == max_length:
            max_len_pred = np.append(max_len_pred, temp_num)


    best_pred = int(stats.mode(max_len_pred)[0])

    print("Best Prediction is :", best_pred)



    cv2.putText(final_image, str(best_pred), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(output_file, final_image)


    return final_image