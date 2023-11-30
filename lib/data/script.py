import os
import sys

cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..', '..'))

from lib.data.prepare_data import makepath, log2file
from lib.data.prepare_data import prepare_vposer_datasets

expr_code = 'version1'

output_datadir = makepath('./data/AMASS/amass_processed/%s' % (expr_code))

logger = log2file(os.path.join(output_datadir, '%s.log' % (expr_code)))
logger('[%s] Preparing data.' % expr_code)

amassx_dir = 'Path of the downloaded AMASS Dataset'
amass_splits = {
    'valid': ['HumanEva', 'HDM05', 'SFU', 'Mosh'],
    'test': ['Transitions', 'SSM'],
    'train': ['CMU', 'PosePrior', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT',
              'BMLrub', 'BMLmovi', 'EKUT', 'TCDHands', 'ACCAD']
}

amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['valid'])))
print(amass_splits)

prepare_vposer_datasets(output_datadir, amass_splits, amassx_dir, logger=logger)