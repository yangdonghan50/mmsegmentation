_base_ = './fcn_hr18_512x1024_160k_acdc.py'
model = dict(
    pretrained='/home/yangdonghan/workspace/mmsegmentation/weights/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))
