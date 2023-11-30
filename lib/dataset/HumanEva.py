import os
import numpy as np
import argparse


def split_npz(npz_fname, output_base_dir, seq_len):
    # Load data
    cdata = np.load(npz_fname, allow_pickle=True)
    fullpose = cdata['poses']
    pose_body = fullpose[:, 3:66]
    root_orient = fullpose[:, :3]

    num_frames = pose_body.shape[0]
    num_batches = num_frames // seq_len

    base_name = os.path.basename(npz_fname).replace('.npz', '')
    subdir = os.path.basename(os.path.dirname(npz_fname))

    output_dir = os.path.join(output_base_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Save each batch as a separate npz file
    for idx in range(num_batches):
        start_idx = idx * seq_len
        end_idx = start_idx + seq_len

        output_name = os.path.join(output_dir, f"{base_name}_batch{str(idx).zfill(3)}.npz")
        np.savez(output_name, pose_body=pose_body[start_idx:end_idx], root_orient=root_orient[start_idx:end_idx])


def main(args):
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.npz'):
                split_npz(os.path.join(root, file), args.output_dir, args.seq_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess pose and trans data and save as npz files.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='The directory where the input .npz files are located.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='The directory where the output .npz files will be saved.')
    parser.add_argument('--seq-len', type=int, default=60,
                        help='Batch size for each .npz file. Default is 60.')

    args = parser.parse_args()
    main(args)
