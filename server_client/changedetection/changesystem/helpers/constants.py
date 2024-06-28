import os
import time
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CHANGEFORMER = 'ChangeFormer'
BIT = "BIT"
CHANGEREX = "ChangerEx"

CHANGE_FORMER_CONFIGS = "changedetection/changesystem/configs/changeformer.json"
BIT_CONFIGS = "changedetection/changesystem/configs/bit.json"
CHANGEREX_CONFIGS = "changedetection/changesystem/configs/changerex.json"
PROJECT_CONFIGS = "changedetection/changesystem/configs/config.json"

RESIZE = "Resize"
CROP = "Crop"
SLIDINGWINDOWAVERAGE = "SlidingWindowAverage"
GAUSSIANSLIDINGWINDOW = "GaussianSlidingWindow"


class AttributeDict(dict):
    """
        dict['key'] -> dict.key converter
        Args: dict (dict): dict to convert
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def count_images_in_directory(dir_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.webp']
    count = 0

    try:
        files = os.listdir(dir_path)
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                count += 1
    except FileNotFoundError:
        print(f"Directory not found: {dir_path}")
    except PermissionError:
        print(f"Permission error accessing: {dir_path}")

    return count


def calculate_average_time(total_time, img_num):
    if img_num == 0:
        return 0.0
    return total_time / img_num

def countTime(predict, path):
    start = time.time()
    predict()
    elapsed = time.time() - start
    img_num = count_images_in_directory(path)
    time_per_one_pair = calculate_average_time(elapsed, img_num)
    return time_per_one_pair
