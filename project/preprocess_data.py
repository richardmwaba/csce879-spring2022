#******************************************************************************************************************#
# Credit goes to https://github.com/hq-jiang/instance-segmentation-with-discriminative-loss-tensorflow. We adopted #
# their code and modified it according to our needs                                                                   #
#******************************************************************************************************************#
import os
import json
import glob
import argparse
from tqdm import tqdm
import cv2
import numpy as np

def read_json(data_dir: str, json_string: str) -> list:
    """Return list of clips from all paths

    Arguments:
        data_dir {str} -- path where data is located
        json_string {str} -- json labels of lanes for each clip

    Returns:
        list -- clips from all paths
    """
    json_paths = glob.glob(os.path.join(data_dir,json_string))
    print (json_paths)
    data = []

    for path in json_paths:
        with open(path) as f:
            d = (line.strip() for line in f)
            d_str = "[{0}]".format(','.join(d))
            data.append(json.loads(d_str))

    num_samples = 0
    for d in data:
        num_samples += len(d)
    print ('Number of labeled images:', num_samples)
    print ('data keys:', data[0][0].keys())
    
    return data

def read_image_strings(data: list, input_dir: str) -> list:
    """Return raw image paths from list of all images

    Arguments:
        data {list} -- list of all images
        input_dir {str} -- path to all initial images

    Returns:
        list -- paths for all raw images
    """
    img_paths = []
    for datum in data:
        for d in datum:
            path = os.path.join(input_dir, d['raw_file'])
            img_paths.append(path)
    
    num_samples = 0
    for d in data:
        num_samples += len(d)
    assert len(img_paths)==num_samples, 'Number of samples do not match'
    print (img_paths[0:2])
    
    return img_paths

def save_input_images(output_dir: str, img_paths: list) -> None:
    """Save all images in one directory

    Arguments:
        output_dir {str} -- target path
        img_paths {list} -- list of paths for all raw images
    """
    for i, path in tqdm(enumerate(img_paths), total=len(img_paths)):
        img = cv2.imread(path)
        output_path = os.path.join(output_dir, '{}.png'.format(str(i).zfill(4)))
        cv2.imwrite(output_path, img)

def draw_lines(img: np.array, lanes: list, height: list, instancewise: bool = False) -> None:
    """Draw lines on given image

    Arguments:
        img {np.array} -- target image
        lanes {list} -- lanes for target image
        height {list} -- height of each lane for target image

    Keyword Arguments:
        instancewise {bool} -- _description_ (default: {False})
    """
    for i, lane in enumerate(lanes):
        pts = [[x,y] for x, y in zip(lane, height) if (x!=-2 and y!=-2)]
        pts = np.array([pts])
        if not instancewise:
            cv2.polylines(img, pts, False,255, thickness=7)
        else:
            cv2.polylines(img, pts, False,50*i+20, thickness=7)

def draw_single_line(img: np.array, lane: list, height: list) -> None:
    """Draw single lane line

    Arguments:
        img {np.array} -- target image
        lane {list} -- lane for target image
        height {list} -- height of lane for target image
    """
    pts = [[x,y] for x, y in zip(lane, height) if (x!=-2 and y!=-2)]
    pts = np.array([pts])
    cv2.polylines(img, pts, False,255, thickness=15)

def save_label_images(output_dir: str, data: list, instancewise: bool =True) -> None:
    """Save labeled images

    Arguments:
        output_dir {str} -- target path
        data {list} -- list of labeled images

    Keyword Arguments:
        instancewise {bool} -- _description_ (default: {True})
    """
    counter = 0

    for i in range(len(data)):
        for j in tqdm(range(len(data[i]))):
            img = np.zeros([720, 1280], dtype=np.uint8)
            lanes = data[i][j]['lanes']
            height = data[i][j]['h_samples']
            draw_lines(img, lanes, height, instancewise)
            output_path = os.path.join(output_dir, '{}.png'.format(str(counter).zfill(4)))
            cv2.imwrite(output_path, img)
            counter += 1


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('srcdir', help="Source directory of TuSimple dataset")
    parser.add_argument('-o', '--outdir', default='.', help="Output directory of extracted data")
    args = parser.parse_args()

    if not os.path.isdir(args.srcdir):
        raise IOError('Directory does not exist')
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    imagesdir = os.path.join(args.outdir, 'images')
    if not os.path.isdir(imagesdir):
        os.mkdir(imagesdir)
    labelsdir = os.path.join(args.outdir, 'labels')    
    if not os.path.isdir(labelsdir):
        os.mkdir(labelsdir)

    json_string = 'label_data_*.json'
    data = read_json(args.srcdir, json_string)
    img_paths = read_image_strings(data, args.srcdir)
    save_input_images(imagesdir, img_paths)
    save_label_images(labelsdir, data)