# **Double Regression Network To Improve Video Super-Resolution**



-- by Feng He( A20479032), XingLi Li( A20482592)

-------

## 1 Introduction

Introduction: including a summary of the problem, previous work, methods, and results.

Problem description: including a detailed description of the problem you try to address methodology.



## 2 Method

Theory: details of technical proof.

Application: detailed description of methods used.



- **Dual Regression Scheme for Paired Data**
- **Dual Regression Scheme for Unpaired Data**



## 3 Experiments

- Paired data test. We mainly use set5, set14, B100, urban100 and manga109 data for this part of the test. From the results, good test results are obtained. In the case of x2 resolution, the contour has been significantly improved. As shown in Figure 3.1 and Figure 3.2 (x2 SR)

![image-20211128215239200](images\image-20211128215239200.png)

Figure 3.1 

![image-20211128215152341](images\image-20211128215152341.png)

Figure 3.2

- This experiment is mainly oriented to the actual data. In practice, it is mainly unpaired data, so we use it for testing.

​    The main process is that we use the third-party video framework to take each frame of the original video as a picture for image super-resolution processing to obtain x2 and X4 super-resolution pictures, and then synthesize each frame of pictures into a new video as the super-resolution video. Some frames are used for comparison.

​    Secondly, we use the whole video as super-resolution and compare the original low resolution video.

![image-20211128212535683](images\image-20211128212535683.png)

Figure 3.3

​     For example, figure 1 is a frame of original image and X2 super-resolution image in the video. It is not particularly obvious, because the contour enhancement is not particularly obvious after zooming in and out according to the scale.

![image-20211128220409914](images\image-20211128220409914.png)

Figure 3.4

​     If you zoom in locally, you can see that the original picture has more granular pixels. As shown in Figure 3.5 and figure 3.6

![image-20211128222714071](images\image-20211128222714071.png)

Figure 3.5

![image-20211128223318186](images\image-20211128223318186.png)

Figure 3.6

## 5 Results

- Problem Description:

​    We found that after super-resolution, the overall good effect on the image is not particularly obvious, but the outline is clear in the specific details. Video is generally dominated by 24 ~ 30 frame rate video. If a frame is used as a picture for super-resolution processing alone, a clearer picture can be obtained. However, due to visual delay and dynamic blur in each frame of video, the improvement of super-resolution on the visual perception of the overall video is relatively limited. As shown in Figure 5.1.

![image-20211128173108114](images\image-20211128173108114.png)

Figure 5.1

​    We analyze the DRN model and think that there are two main reasons why DRN is not ideal in video super-resolution.

​    First, the picture quality of each frame of video is not as good as that of a single picture. In addition to the problem of low resolution, there are also dynamic blurred photos and blur caused by depth of field light and shadow;

​    Second, DRN model is a model trained from ordinary pictures. There is no good fitting for the processing of blurred pictures in video and  various light and shadow pictures.

​    For these two reasons, we think that using video to train DRN model may be better. And this will greatly increase the amount of training data.

- Processing time

​     This reason is inevitable in real problems. In the test, we use a 480x270 resolution 24/s frame rate 37 second video for test, which takes about 30 minutes. If the resolution is greater, it will cause the memory overflow problem of 4G graphics card. As shown in Figure 5.2, more than 70% of the time spent on super-resolution of 50 frames 480x270 pictures is actually in the process of image generation. In terms of time and memory consumption of graphics card, it is not very friendly.

![image-20211128210306337](images\image-20211128210306337.png)

Figure 5.2

## 6 Future Work

   From the experimental results, we know that DRN can significantly improve a single picture. In each frame of video processing, model processing consumes more time, accounting for more than 70% of the processing time. In the future work, we hope to improve the overall improvement of video super-resolution and image generation performance of DRN.

## 7 Conclusion

Conclusions and future work: including a brief summary of the main contributions of the project and the lessons you learn from the project, as

well as a list of some potential future work.



## 7 References