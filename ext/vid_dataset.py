import os

from torch.utils.data import Dataset


class video_read(Dataset):
    def __init__(self, videofolder):
        self.videofolder = videofolder
        self.videos = os.listdir(videofolder)
        self.__check_folder()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        name = self.videos[index]
        name = os.path.join(self.videofolder, name)
        return name

    def __check_folder(self):
        if not os.path.exists("./output/"):
            os.mkdir("./output/")
        if not os.path.exists("./temp/"):
            os.mkdir("./temp/")
