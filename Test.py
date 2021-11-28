# -*- coding: utf-8 -*-
"""
Create Time: 2021/11/26 21:48
Author: Eric
Desc：Todo
"""
import cv2

from ext.vid_core import video_SR

if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_SR('./input/', fourcc)
