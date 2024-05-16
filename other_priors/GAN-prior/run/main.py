import sys
import os

import pprint

_root_path = os.path.join(os.path.dirname(__file__) , '..')
sys.path.insert(0, _root_path)

import lib

def main():

    ### initialize Trainer
    cfg = lib.utils.utils.parse_args().cfg
    trainer = lib.core.trainer.Trainer(cfg)

    ### copy yaml description file to the save folder
    lib.utils.utils.copy_exp_file(trainer)

    ### copy proc.py file to the save folder
    lib.utils.utils.copy_proc_file(trainer)

    trainer.logger.info(pprint.pformat(trainer.cfg))
    trainer.logger.info('#'*100)

    ### run the training procedure
    trainer.run()


if __name__ == '__main__':
    main()