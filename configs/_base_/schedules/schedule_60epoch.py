# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(by_epoch=True, interval=30)
evaluation = dict(interval=30, metric='mIoU', pre_eval=True)
