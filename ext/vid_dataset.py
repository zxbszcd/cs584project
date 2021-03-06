import os

from torch.utils.data import Dataset

from ext import check_file_format


class video_read(Dataset):
    def __init__(self, videofolder):
        self.videofolder = videofolder
        videos = []
        for item in self.getAllFiles(videofolder):
            name, format = os.path.splitext(item)
            format = format.replace(".", "")
            if (format in check_file_format.video_format_list):
                videos.append(item)
        self.videos = videos
        self.__check_folder()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        name = self.videos[index]
        return name

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
