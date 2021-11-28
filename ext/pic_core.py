# -*- coding: utf-8 -*-
"""
Create Time: 2021/11/28 17:50
Author: Eric
Desc：Todo
"""
import os

import cv2
import torch

import data
import model
from checkpoint import Checkpoint
from ext.pic_dataset import picture_read
from ext.vid_core import del_temp
from option import args
from trainer import Trainer

checkpoint = Checkpoint(args)


def pic_SR(picFolder):
    del_temp()  # Clean up the temporary folder first
    data_num = picture_read(picFolder).__len__()
    loader = data.Data(args)
    model_net = model.Model(args, checkpoint)
    t = Trainer(args, loader, model_net, None, checkpoint)
    print("Total: " + str(data_num) + " number pictures")
    for i in range(data_num):
        print("Processing " + str(i + 1) + "-th picture", end='\t')
        img, file_path = picture_read(picFolder).__getitem__(i)
        image_SR2, image_SR4 = t.transform_picture(file_path)
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_name, format = os.path.splitext(file_name)
        output_path = file_dir.replace("input", "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(f'{output_path}/{file_name}x2{format}', image_SR2)
        cv2.imwrite(f'{output_path}/{file_name}x4{format}', image_SR4)
        print("Finish！")
        torch.cuda.empty_cache()
