import cv2
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def process(HOME_DIR, json_filename, SIZE=256):
    """Convert TuSimple json labels to binary masks
    Argument:
        HOME_DIR: path to TuSimple dataset
        json_filename: either 'label_data_0313.json' 'label_data_0531.json', or 'label_data_0601.json'
        SIZE: figure size of the processed images and masks
    Returns:
        None
    """
    json_gt = [json.loads(line) for line in open(HOME_DIR + json_filename)]  # there are 3 
    
    # specify paths to store resized images and binary masks
    resized_img_path = os.path.join(HOME_DIR, 'original_image')
    gt_mask_path = os.path.join(HOME_DIR, 'label_image')
    if not os.path.isdir(resized_img_path):
        os.makedirs(resized_img_path, exist_ok=True)
    if not os.path.isdir(gt_mask_path):
        os.makedirs(gt_mask_path, exist_ok=True)

    for i, gt in enumerate(json_gt):

        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']

        img = cv2.imread(HOME_DIR + raw_file)

        gt_lanes_label = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]

        img_label = np.zeros(img.shape)

        for lane in gt_lanes_label:
            cv2.polylines(img_label, np.int32([lane]), isClosed=False, color=(1, 1, 1), thickness=5)

        # resize img and mask
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        img_label = cv2.resize(img_label, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)

        generated_file = "_".join(raw_file.split('/')[:-1]) + ".png"

        plt.imsave(os.path.join(resized_img_path, generated_file), img)
        plt.imsave(os.path.join(gt_mask_path, generated_file), img_label)
        

def mask2json():  # Convert binary prediction masks to json (NOT WORKING)
    train_outputs = []
    val_outputs = []
    test_outputs = []
    for count, gt in enumerate(json_gt):
        gt_lanes = gt['lanes']
        raw_file = gt['raw_file']
        generated_file = "_".join(raw_file.split('/')[:-1]) + ".png"
        print(generated_file)

        if generated_file not in test_file_name: continue

        # load the image, convert it to grayscale, blur it slightly,
        # and threshold it

        #print (args["home_dir"] )
        image = cv2.imread(pred_mask_path + generated_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)[1]

        output = {}
        output["lanes"] = [[] for _ in range(len(gt_lanes))]
        output["h_samples"] = []
        output["raw_file"] = raw_file
        for h_start in range(155, 715, 10):
            output["h_samples"].append(h_start + 5)

            slice = thresh[h_start: h_start + 10, :]

            # find contours in the thresholded image
            cnts = cv2.findContours(slice.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            #loop over the contours
            for i in range(len(gt_lanes)):
                print(i+1)
                if i < len(cnts):
                    # compute the center of the contour
                    c = cnts[i]
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        print(cX)
                        cY = int(M["m01"] / M["m00"])

                        # draw the contour and center of the shape on the image
                        cv2.circle(image, (cX, h_start + 5), 3, (255, 255, 255), -1)
                        output["lanes"][i].append(cX)
                    else:
                        output["lanes"][i].append(-2)
                else:
                    output["lanes"][i].append(-2)

        test_outputs.append(output)

    with open('test_pred.txt', 'w') as outfile:
        for output in test_outputs:
            json.dump(output, outfile)
            outfile.write("\n")
    outfile.close()
        
        
def plot_performance(results, metrics=['accuracy', 'loss']):
    """Plot performance metrics
    Argument:
        results: training results from Keras's model.fit()
        metrics: performance metrics for trained model
    Returns:
        plt figure
    """
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(results.history[metrics[0]])
    plt.plot(results.history['val_'+metrics[0]], '')
    plt.xlabel("Epochs")
    plt.ylabel(metrics[0])
    plt.legend([metrics[0], 'val_'+metrics[0]])
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plt.plot(results.history[metrics[1]])
    plt.plot(results.history['val_'+metrics[1]], '')
    plt.xlabel("Epochs")
    plt.ylabel(metrics[1])
    plt.legend([metrics[1], 'val_'+metrics[1]])
    plt.ylim(0, None)
    plt.show()