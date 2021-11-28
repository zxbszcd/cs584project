# Super resolution video processing based on DRN

## Dependencies
```
Python>=3.7, PyTorch>=1.1, numpy, skimage, imageio, moviepy, matplotlib, tqdm
```
## Model Testing

| Model | Scale | #Params (M) | PSNR on Set5 (dB) |
| :---: | :---: | :---------: | :---------------: |
| DRN-S |   4   |     4.8     |       32.68       |

Put the test video into the input directory, and X2 and X4 videos will be generated in the output directory.

For example, use the following command to test our DRN-S model for 4x SR.

```
python Test.py --scale 4 --model DRN-S --pre_train premodel/DRNS4x.pt --test_only
```

If you want to load the pre-trained dual model, you can add the following option into the command.

```
python Test.py --scale 4 --model DRN-S --pre_train premodel/DRNS4x.pt --pre_train_dual premodel/DRNS4x_dual_model.pt  --test_only
```

Citation

```
@inproceedings{guo2020closed,
  title={Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution},
  author={Guo, Yong and Chen, Jian and Wang, Jingdong and Chen, Qi and Cao, Jiezhang and Deng, Zeshuai and Xu, Yanwu and Tan, Mingkui},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```