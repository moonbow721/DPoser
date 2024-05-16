"""Generate script for creating training data from AMASS
python data/prepare_data.py --raw_data /BS/humanpose/static00/data/PoseNDF_raw/smpl_h --out_path /BS/humanpose/static00/data/PoseNDF_train/smpl_h --bash_file ./traindata.sh --use_slrum
python data/prepare_data.py --raw_data <path_for_samples_amass_poses> --out_path <path_for_training_data> --bash_file ./traindata.sh 
 
 Intrsuctions for running the code:
 Please change L24:L30, if you are using slurm """

import argparse
import os

import numpy as np
import ipdb

from data_splits import amass_splits

def main(args):
    bash_path = args.bash_file
    amass_datas = sorted(amass_splits['train'])
    print(amass_datas)
    with open(bash_path, 'w+') as fp:

        if args.use_slurm:
            # ToDo: change these paths accordingly
            fp.write("#!/bin/bash"+ "\n")
            fp.write("#SBATCH -p gpu20"+ "\n")
            fp.write("#SBATCH --signal=B:SIGTERM@120"+ "\n")
            fp.write("#SBATCH -c 5"+ "\n")
            fp.write("#SBATCH --mem-per-cpu=6144"+ "\n")
            fp.write("#SBATCH --gres gpu:1"+ "\n")
            fp.write("#SBATCH -t 0-12:00:00"+ "\n")
            fp.write("#SBATCH -a 1-409%409"+ "\n")
            fp.write('#SBATCH -o "/scratch/inf0/user/gtiwari/slurm-%A.out"'+ "\n")
            fp.write('#SBATCH -e "/scratch/inf0/user/gtiwari/slurm-%A.err"'+ "\n")
            fp.write("#SBATCH --gres gpu:1"+ "\n")
            fp.write("source /BS/garvita/static00/software/miniconda3/etc/profile.d/conda.sh"+ "\n")
            fp.write("conda activate posendf"+ "\n")
            fp.write("cd /BS/garvita/work/code/PoseNDF "+ "\n")
            fp.write("case $SLURM_ARRAY_TASK_ID in" + "\n")
        count = 1
        out_dir = args.out_path
        for amass_data in amass_datas:
            ds_dir = os.path.join(args.raw_data,amass_data)
            seqs = sorted(os.listdir(ds_dir))
            for seq in seqs:
                if  os.path.exists(os.path.join(out_dir, amass_data, seq)):
                    print('done....', amass_data,  seq)
                    continue

                if not 'npz' in seq:
                    continue
                if args.use_slurm:
                    fp.write("\t {})".format(count) + "\n")
                    fp.write("\t\t\t")
                fp.write(
                    "python data/prepare_traindata.py --seq_file {}/{} --runs {} --out_dir {}".format(amass_data, seq, 1, out_dir))
              
                count += 1
                if args.use_slurm:
                    fp.write( "& \n")
                    fp.write("\t\t\t;;\n")
                else:
                    fp.write( "\n")
        if args.use_slurm:
            fp.write("esac"+ "\n\n" + "wait")
        print("Total sequences to be processed....", count)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for creating PoseNDF dataset")

    # Paths to output files
    parser.add_argument("-rd", "--raw_data", type=str, default="/BS/humanpose/static00/data/PoseNDF_raw/smpl_h2",
                        help="Path to the sampled poses from AMASS")
    parser.add_argument("-op", "--out_path", type=str, default="/BS/humanpose/static00/data/PoseNDF_train/smpl_h_small",
                        help="Path to the resulting datafolder(storing dataset)")
    parser.add_argument("-bf", "--bash_file", type=str, default="./tmp.sh",
                        help="Path to the bash script file")
    parser.add_argument('-sl', '--use_slurm',  action="store_true", help="Using slurm for creating dataset")
    arguments = parser.parse_args()
    main(arguments)

   