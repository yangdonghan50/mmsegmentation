import torch
import os
import argparse
import shutil
import sys
import numpy as np
import pickle
from collections import OrderedDict


def convert_from_txt(ori_model, ori_txt, dst_model, dst_txt, save_path):
    ori_key_list = open(ori_txt, 'r').readlines()
    dst_key_list = open(dst_txt, 'r').readlines()
    assert len(ori_key_list) == len(dst_key_list)

    if ori_model.endswith('.pkl'):
        f = open(ori_model, 'rb')
        ori_model = pickle.load(f)['model']
    else:
        try:
            ori_model = torch.load(ori_model, map_location=torch.device('cpu'))
        except:
            ori_model = torch.load(ori_model, map_location=torch.device('cpu'))

    if dst_model.endswith('.pkl'):
        f = open(dst_model, 'rb')
        dst_model = pickle.load(f)['model']
    else:
        try:
            dst_model = torch.load(dst_model, map_location=torch.device('cpu'))
        except:
            dst_model = torch.load(dst_model, map_location=torch.device('cpu'))

    new_model = {}
    new_model['meta'] = ori_model['meta']
    new_model['state_dict'] = {}
    for i in range(len(ori_key_list)):
        ori_k = ori_key_list[i].split('  ')[0]
        dst_k = dst_key_list[i].split('  ')[0]
        # ori_model[ori_k] = torch.from_numpy(dst_model[dst_k])
        assert ori_model['state_dict'][ori_k].shape == dst_model['state_dict'][dst_k].shape
        new_model['state_dict'][ori_k] = dst_model['state_dict'][dst_k]
        print(ori_k, dst_k, ori_model['state_dict'][ori_k].shape, dst_model['state_dict'][dst_k].shape)

    torch.save(new_model, save_path)
    # save_path = open(save_path, 'wb')
    # model = {}
    # model['model'] = ori_model
    # pickle.dump(model, save_path)


if __name__ == '__main__':
    dst_model = '/home/yangdonghan/workspace/best_checkpoint_86.76_PSA_s.pth'
    dst_txt = '/home/yangdonghan/workspace/official_pas_backbone.txt'
    ori_model = '/home/yangdonghan/workspace/mmsegmentation/work_dirs/ocrnet_hr48_576x1024_60epoch_acdc/iter_100.pth'
    ori_txt = '/home/yangdonghan/workspace/mm_psa_basckbone.txt'
    save_path = '/home/yangdonghan/workspace/mmsegmentation/weights/converted_psa_backbone.pth'
    convert_from_txt(ori_model, ori_txt, dst_model, dst_txt, save_path)
