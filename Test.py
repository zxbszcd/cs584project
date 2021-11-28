# -*- coding: utf-8 -*-
"""
Create Time: 2021/11/26 21:48
Author: Eric
Desc：Todo
"""
import cProfile
import pstats

import cv2

from ext.pic_core import pic_SR
from ext.vid_core import video_SR


def test_video():
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_SR('./input/', fourcc)


def test_picture():
    pic_SR('./input/')


def show_time():
    p = pstats.Stats('restats')
    # p.strip_dirs().sort_stats(-1).print_stats()
    p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    p.sort_stats(pstats.SortKey.TIME).print_stats(10)


if __name__ == '__main__':
    # test_video()
    # test_picture()

    cProfile.run('test_video()', 'restats')
    show_time()
    # cProfile.run('test_picture()', 'restats')
    # show_time()
