import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy
import torch
from moviepy.editor import VideoFileClip

import data
from ext.br import get_flag
from checkpoint import Checkpoint
from option import args
from trainer import Trainer
from ext.vid_dataset import video_read

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
        name = video_read(videoFolder).__getitem__(n)
        cap = cv2.VideoCapture(name)
        frame_num = int(cap.get(7))
        weight = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(5)

        output_filepath = name.replace('input', 'temp')
        output_filepath, video_format = os.path.splitext(output_filepath)
        output_filepath2 = output_filepath + '_x2' + video_format
        output_filepath4 = output_filepath + '_x4' + video_format

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
        video_clip = VideoFileClip(name)
        videoClip2 = videoClip2.set_audio(video_clip.audio) # Set audio for video
        filepath2 = output_filepath2.replace('temp', 'output')
        videoClip2.write_videofile(filepath2)

        videoClip4 = VideoFileClip(output_filepath4)
        video_clip = VideoFileClip(name)
        videoClip4 = videoClip4.set_audio(video_clip.audio)
        filepath4 = output_filepath4.replace('temp', 'output')
        videoClip4.write_videofile(filepath4)

        current = datetime.now()
        print(f"end time：{current}")

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
