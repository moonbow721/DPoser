# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.01.02

import glob
import os.path as osp
import shutil
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)

save_betas = True
num_betas = 10

def logger_sequencer(logger_list, prefix=None):
    def post_text(text):
        if prefix is not None: text = '{} -- '.format(prefix) + text
        for logger_call in logger_list: logger_call(text)

    return post_text


class log2file():
    def __init__(self, logpath=None, prefix='', auto_newline=True, write2file_only=False):
        if logpath is not None:
            makepath(logpath, isfile=True)
            self.fhandle = open(logpath, 'a+')
        else:
            self.fhandle = None

        self.prefix = prefix
        self.auto_newline = auto_newline
        self.write2file_only = write2file_only

    def __call__(self, text):
        if text is None: return
        if self.prefix != '': text = '{} -- '.format(self.prefix) + text
        # breakpoint()
        if self.auto_newline:
            if not text.endswith('\n'):
                text = text + '\n'
        if not self.write2file_only: sys.stderr.write(text)
        if self.fhandle is not None:
            self.fhandle.write(text)
            self.fhandle.flush()


def makepath(*args, **kwargs):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    isfile = kwargs.get('isfile', False)
    import os
    desired_path = os.path.join(*args)
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


def dataset_exists(dataset_dir, split_names=None):
    '''
    This function checks whether a valid SuperCap dataset directory exists at a location
    Parameters
    ----------
    dataset_dir

    Returns
    -------

    '''
    if dataset_dir is None: return False
    if split_names is None:
        split_names = ['train', 'valid', 'test']
    import os

    import numpy as np

    done = []
    for split_name in split_names:
        checkfiles = ['root_orient', 'pose_body']
        if save_betas:
            checkfiles = checkfiles + ['betas']
        for k in checkfiles:
            outfname = os.path.join(dataset_dir, split_name, f'{k}.pt')
            done.append(os.path.exists(outfname))
    return np.all(done)


def prepare_vposer_datasets(vposer_dataset_dir, amass_splits, amass_dir, logger=None):
    if dataset_exists(vposer_dataset_dir):
        if logger is not None: logger(f'VPoser dataset already exists at {vposer_dataset_dir}')
        return

    ds_logger = log2file(makepath(vposer_dataset_dir, 'dataset.log', isfile=True), write2file_only=True)
    logger = ds_logger if logger is None else logger_sequencer([ds_logger, logger])

    logger(f'Creating pytorch dataset at {vposer_dataset_dir}')
    logger(f'Using AMASS body parameters from {amass_dir}')

    shutil.copy2(__file__, vposer_dataset_dir)

    def fetch_from_amass(ds_names):
        keep_rate = 0.3

        npz_fnames = []
        for ds_name in ds_names:
            # mosh_stageII_fnames = glob.glob(osp.join(amass_dir, ds_name, '*/*_poses.npz'))
            mosh_stageII_fnames = glob.glob(osp.join(amass_dir, ds_name, '*/*_stageii.npz'))
            npz_fnames.extend(mosh_stageII_fnames)
            logger('Found {} sequences from {}.'.format(len(mosh_stageII_fnames), ds_name))

            for npz_fname in npz_fnames:
                print(npz_fname)
                cdata = np.load(npz_fname, allow_pickle=True)
                N = len(cdata['poses'])
                # skip first and last frames to avoid initial standard poses, e.g. T pose
                cdata_ids = np.random.choice(list(range(int(0.1 * N), int(0.9 * N), 1)), int(keep_rate * 0.8 * N),
                                             replace=False)
                if len(cdata_ids) < 1: continue
                fullpose = cdata['poses'][cdata_ids].astype(np.float32)
                result_dict = {'pose_body': fullpose[:, 3:66], 'root_orient': fullpose[:, :3]}
                if save_betas:
                    result_dict['betas'] = np.tile(cdata['betas'][:num_betas], (len(cdata_ids), 1))

                yield result_dict

    for split_name, ds_names in amass_splits.items():
        if dataset_exists(vposer_dataset_dir, split_names=[split_name]): continue
        logger(f'Preparing VPoser data for split {split_name}')

        data_fields = {}
        for data in fetch_from_amass(ds_names):
            for k in data.keys():
                if k not in data_fields: data_fields[k] = []
                data_fields[k].append(data[k])

        for k, v in data_fields.items():
            outpath = makepath(vposer_dataset_dir, split_name, '{}.pt'.format(k), isfile=True)
            v = np.concatenate(v)
            torch.save(torch.tensor(v), outpath)

        logger(
            f'{len(v)} datapoints dumped for split {split_name}. ds_meta_pklpath: {osp.join(vposer_dataset_dir, split_name)}')

    logger(f'Dumped final pytorch dataset at {vposer_dataset_dir}')
