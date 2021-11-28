# Super resolution video processing based on DRN

- Test effect method, execute command

Put the test video into the input directory, and X2 and X4 videos will be generated in the output directory.

```
python Test.py --scale 4 --model DRN-S --pre_train premodel/DRNS4x.pt --pre_train_dual premodel/DRNS4x_dual_model.pt --test_only
```

