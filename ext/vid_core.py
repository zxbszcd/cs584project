import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy
import torch
from moviepy.editor import VideoFileClip

import data
from checkpoint import Checkpoint
from ext.br import get_flag
from ext.vid_dataset import video_read
from option import args
from trainer import Trainer

checkpoint = Checkpoint(args)
import model


def video_SR(videoFolder, fourcc):
    del_temp()  # Clean up the temporary folder first
    data_num = video_read(videoFolder).__len__()
    loader = data.Data(args)
    model_net = model.Model(args, checkpoint)
    t = Trainer(args, loader, model_net, None, checkpoint)
    print("Total: " + str(data_num) + "number video")
    for n in range(data_num):
        print("Processing " + str(n + 1) + "-th video")
        file_path = video_read(videoFolder).__getitem__(n)
        cap = cv2.VideoCapture(file_path)
        frame_num = int(cap.get(7))
        # frame_num = 50
        weight = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(5)
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_name, video_format = os.path.splitext(file_name)
        output_dir = file_dir.replace("input", "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_filepath2 = output_dir + "/" + file_name + '_x2' + video_format
        output_filepath4 = output_dir + "/" + file_name + '_x4' + video_format

        scale2 = 2
        scale4 = 4
        out2 = cv2.VideoWriter(output_filepath2, fourcc, fps, (weight * scale2, height * scale2), True)
        out4 = cv2.VideoWriter(output_filepath4, fourcc, fps, (weight * scale4, height * scale4), True)
        i = 0
        current = datetime.now()
        print(f"start time：{current}")
        while cap.isOpened():
            if get_flag():
                break
            ret, frame = cap.read()
            print("The video has a total of " + str(frame_num) + " frames，Processing " + str(i) + "frame", end="\r")
            frame = numpy.array(frame, dtype='float32')
            cv2.imwrite(f'./temp/tmp.bmp', frame)
            if frame.size == 1:
                print("\nProcessing frame. This frame is empty and will exit")
                break
            frame_SR2, frame_SR4 = t.transform_frame(frame)
            cv2.imwrite(f'./temp/tmp_x2.bmp', frame_SR2)
            cv2.imwrite(f'./temp/tmp_x4.bmp', frame_SR4)
            frame_SR2 = cv2.imread('./temp/tmp_x2.bmp')
            frame_SR4 = cv2.imread('./temp/tmp_x4.bmp')
            out2.write(frame_SR2)
            out4.write(frame_SR4)
            i += 1
            if i == frame_num:
                break

        cap.release()
        out2.release()
        out4.release()

        videoClip2 = VideoFileClip(output_filepath2)
        video_clip = VideoFileClip(file_path)
        videoClip2 = videoClip2.set_audio(video_clip.audio)  # Set audio for video
        videoClip2.write_videofile(output_filepath2)

        videoClip4 = VideoFileClip(output_filepath4)
        video_clip = VideoFileClip(file_path)
        videoClip4 = videoClip4.set_audio(video_clip.audio)
        videoClip4.write_videofile(output_filepath4)

        print(f"take time：{datetime.now() - current}")

        # del_temp()
        print("Finish!")
        torch.cuda.empty_cache()


def del_temp():
    tmp_file = Path("./temp")
    if tmp_file.exists():
        del_list = os.listdir('./temp')  #
        for file in del_list:
            file_path = os.path.join('./temp/', file)
            if os.path.isfile(file_path):
                os.remove(file_path)
