
amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje' ,'BMLhandball' ,
              'BMLmovi', 'EKUT', 'TCD_handMocap',
              'ACCAD']
}
amass_splits['train'] = list(
    set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))
