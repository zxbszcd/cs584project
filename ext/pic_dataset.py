import os

import cv2
import numpy
import torch
from torch.utils.data import Dataset

from ext import check_file_format


class picture_read(Dataset):
    def __init__(self, imageFolder):
        self.imageFolder = imageFolder
        prictures = []
        for item in self.getAllFiles(imageFolder):
            name, format = os.path.splitext(item)
            format = format.replace(".", "")
            if (format in check_file_format.picture_format_list):
                prictures.append(item)
        self.images = prictures
        self.__check_folder()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image = cv2.imdecode(numpy.fromfile(name, dtype=numpy.uint8), -1)
        image = numpy.array(image, dtype='float32')
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image, name

    def __check_folder(self):
        if not os.path.exists("./output/"):
            os.mkdir("./output/")
        if not os.path.exists("./temp/"):
            os.mkdir("./temp/")

    def getAllFiles(self, targetDir):
        files = []
        listFiles = os.listdir(targetDir)
        for i in range(0, len(listFiles)):
            path = os.path.join(targetDir, listFiles[i])
            if os.path.isdir(path):
                files.extend(self.getAllFiles(path))
            elif os.path.isfile(path):
                files.append(path)
        return files
