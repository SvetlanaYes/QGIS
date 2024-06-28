import torch
import json
import torchvision.transforms as T
from .helpers.gaussian_helpers import set_generators, apply_window_prediction
from .helpers.constants import DEVICE
import os
# from helpers.gaussian_helpers import set_generators, apply_window_prediction
# from helpers.constants import DEVICE
file_path = 'changedetection/changesystem/configs/config.json'
absolute_path = os.path.abspath(file_path)
with open(absolute_path) as f:
    config = json.load(f)


class Methods:
    def __init__(self, window_size=None, stride=None, sigma=None):
        self.window_size = window_size
        self.stride = stride
        self.sigma = sigma

    def process(self, batch):
        image1 = batch[config["first_image_dir"]]
        image2 = batch[config["second_image_dir"]]
        height, width = image1.shape[2], image1.shape[3]
        cropped_images = []
        flag0 = ((height - self.window_size) % self.stride) != 0
        flag1 = ((width - self.window_size) % self.stride) != 0
        for i in range(0, height-self.window_size+1, self.stride):
            i_finish = i+self.window_size
            for j in range(0, width-self.window_size+1, self.stride):                    
                j_finish = j+self.window_size
                cropped_images.append((image1[:, :, i:i_finish, j:j_finish], image2[:, :, i:i_finish, j:j_finish]))
            if flag1:
                j = width - self.window_size
                cropped_images.append((image1[:, :, i:i_finish, j:], image2[:, :, i:i_finish, j:]))
        if flag0:
            i = height - self.window_size
            for j in range(0, width-self.window_size+1, self.stride):
                j_finish = j+self.window_size
                cropped_images.append((image1[:, :, i:, j:j_finish], image2[:, :, i:, j:j_finish]))
            if flag1:
                j = width - self.window_size
                cropped_images.append((image1[:, :, i:, j:], image2[:, :, i:, j:]))
        return cropped_images

    def merge(self, predictions_generator, image_shape=(8, 2, 256, 256)):
        _, _, height, width = image_shape
        result = torch.zeros(image_shape).to(DEVICE)
        count_matrix = torch.zeros(image_shape).to(DEVICE)
        flag0 = ((height - self.window_size) % self.stride) != 0
        flag1 = ((width - self.window_size) % self.stride) != 0
        for i in range(0, height-self.window_size+1, self.stride):
            i_finish = i+self.window_size
            for j in range(0, width-self.window_size+1, self.stride):                    
                j_finish = j+self.window_size
                result[:, :, i:i_finish, j:j_finish] += next(predictions_generator).to(DEVICE)
                count_matrix[:, :, i:i_finish, j:j_finish] += 1
            if flag1:
                j = width - self.window_size
                result[:, :, i:i_finish, j:] += next(predictions_generator).to(DEVICE)
                count_matrix[:, :, i:i_finish, j:] += 1
        if flag0:
            i = height - self.window_size
            for j in range(0, width-self.window_size+1, self.stride):
                j_finish = j+self.window_size
                result[:, :, i:, j:j_finish] += next(predictions_generator).to(DEVICE)
                count_matrix[:, :, i:, j:j_finish] += 1
            if flag1:
                j = width - self.window_size
                result[:, :, i:, j:] += next(predictions_generator).to(DEVICE)
                count_matrix[:, :, i:, j:] += 1
        result /= count_matrix
        return result



class Resize(Methods):
    def __init__(self, window_size=256, stride=None, sigma=None):
        super().__init__(window_size=window_size, stride=None, sigma=None)

    def process(self, batch):
        image1 = batch[config["first_image_dir"]]
        image2 = batch[config["second_image_dir"]]
        image1 = image1 / 2 + 0.5 
        image2 = image2 / 2 + 0.5
        label_exists = (config["label_dir"] in batch)
        if label_exists:
            label = batch[config["label_dir"]]
        
        target_shape = (self.window_size, self.window_size)
        to_pil = T.ToPILImage()

        image1_batch = [to_pil(img) for img in image1]
        image2_batch = [to_pil(img) for img in image2]
        if label_exists:
            label_batch = [to_pil(img) for img in label]
        resize = T.Resize(target_shape)
        image1_resized_batch = [resize(img) for img in image1_batch]
        image2_resized_batch = [resize(img) for img in image2_batch]
        if label_exists:
            label_resized_batch = [resize(img) for img in label_batch]
        to_tensor = T.ToTensor()
        image1_batch = torch.cat([to_tensor(img).unsqueeze(0) for img in image1_resized_batch], dim=0)
        image2_batch = torch.cat([to_tensor(img).unsqueeze(0) for img in image2_resized_batch], dim=0)
        image1_batch = (image1_batch - 0.5) * 2 
        image2_batch = (image2_batch - 0.5) * 2
        if label_exists:
            label_batch = torch.cat([to_tensor(img).unsqueeze(0) for img in label_resized_batch], dim=0) * 255
            batch[config["label_dir"]] = label_batch.to(torch.int).to(DEVICE)
        return [(image1_batch, image2_batch)]

    def merge(self, predictions_generator, _):
        return next(predictions_generator)


class Crop(Methods):
    def __init__(self, window_size=256, stride=None, sigma=None):
        super().__init__(window_size=window_size, stride=stride, sigma=sigma)
        self.stride = window_size


class SlidingWindowAverage(Methods):
    def __init__(self, window_size=256, stride=64, sigma=None):
        super().__init__(window_size=window_size, stride=stride, sigma=None)


class GaussianSlidingWindow(Methods):
    def __init__(self,window_size=256, stride=64, sigma=100):
        super().__init__(window_size=window_size, stride=stride, sigma=sigma)
        self.generators = None

    def merge(self, predictions_generator, image_shape=(8, 2, 256, 256)):
        if self.generators is None:
            self.generators = set_generators(self.window_size, self.stride, self.sigma, image_shape[2:])

        _, _, height, width = image_shape
        result = torch.zeros(image_shape).to(DEVICE)
        flag0 = ((height - self.window_size) % self.stride) != 0  
        flag1 = ((width - self.window_size) % self.stride) != 0
        for i in range(0, height-self.window_size+1, self.stride):
            for j in range(0, width-self.window_size+1, self.stride):
                predict_window = next(predictions_generator).to(DEVICE)
                apply_window_prediction(result, predict_window, self.generators, i, j, height, width)
            if flag1:
                j = width - self.window_size
                predict_window = next(predictions_generator).to(DEVICE)
                apply_window_prediction(result, predict_window, self.generators, i, j, height, width)
        if flag0:
            i = height - self.window_size
            for j in range(0, width-self.window_size+1, self.stride):
                predict_window = next(predictions_generator).to(DEVICE)
                apply_window_prediction(result, predict_window, self.generators, i, j, height, width)
            if flag1:
                j = width - self.window_size
                predict_window = next(predictions_generator).to(DEVICE)
                apply_window_prediction(result, predict_window, self.generators, i, j, height, width)
        return result

