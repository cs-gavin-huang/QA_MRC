'''
Author: geekli
Date: 2020-12-28 15:28:31
LastEditTime: 2020-12-28 15:39:51
LastEditors: your name
Description: 
FilePath: \QA_MRC\best_cn_finetune\utils.py
'''
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections
import re
import torch
from glob import glob

def check_args(args,rank=0):
    args.setting_file = os.path.join(args.checkoint_dir,args.setting_file)
    args.log_file = os.path.join(args.checkoint_dir,args.log_file)
    #args add log_file and setting_file
    if rank == 0:
        os.makedirs(args.checkoint_dir, exist_ok=True)
        with open(args.setting_file, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            print('------------ Options -------------')
            for k in args.__dict__:
                v = args.__dict__[k]
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            print('------------ End -------------')

    return args

def show_all_variables(rank=0):
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars,print_info=True if rank == 0 else False)

def torch_show_all_params(model, rank=0):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    if rank == 0:
        print("Totel param num:" + str(k))

#import import ipdb; ipdb.set_trace()
def get_assignment_map_from_checkpoint(tvars,init_checkpoint):
    """compute the union of the current variables and checkoint variables"""
    initialized_variable_names = {}
    new_variable_names = set()
    unused_variable_names = set()

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            if 'adam' not in name:
                unused_variable_names.add(name)
            continue
        # assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    for name in name_to_variable:
        if name not in initialized_variable_names:
            new_variable_names.add(name)
    return assignment_map, initialized_variable_names, new_variable_names, unused_variable_names

