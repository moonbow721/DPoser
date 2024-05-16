""""Prepare training data"""

import os
import numpy as np
import pytorch3d
import faiss
import ipdb
from data_splits import amass_splits
import argparse
import torch
from create_data import PoseData, quat_doublecover, quat_flip, axis_angle_to_quaternion_np, AmassData
import faiss
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
import faiss.contrib.torch_utils
import dist_utils
import smplx


def faiss_idx_np_jts(amass_datas, root_dir, bm, device='cuda:0', batch_size=128):
    all_joints = []
    all_pose = []
    print(len(amass_datas))
    tmp = 0

    betas = torch.zeros((batch_size,10)).to(device=device)
    print(amass_datas)

    for amass_data in amass_datas:
        ds_dir = os.path.join(root_dir,amass_data)
        seqs = sorted(os.listdir(ds_dir))
        print(amass_data, len(seqs))


        for seq in seqs:
            if not 'npz' in seq:
                continue
            amass_seq = AmassData(os.path.join(ds_dir, seq),  batch_size=batch_size, num_workers=3)
            amass_seq_dataloader = amass_seq.get_loader()
            for query_batch in amass_seq_dataloader:
                amass_pose = query_batch.get('pose').to(device=device)
                body_out =bm( betas=betas,body_pose=amass_pose)
                all_joints.extend(body_out.joints.detach().cpu().numpy()[:, :25].reshape(batch_size, -1))
                all_pose.extend(amass_pose.detach().cpu().numpy().reshape(batch_size, -1))

            # print(seq)
    ipdb.set_trace()
    print('total size....', tmp/(1024*1024*1024))
    all_pose = np.array(all_pose)
    all_joints = np.array(all_joints)
    # index = faiss.index_factory(84, "Flat",faiss.METRIC_INNER_PRODUCT)
    index = faiss.index_factory(75, "Flat")
    print('training faiss')

    index.train(all_joints)
    print('trained faiss')

    index.add(all_joints)
    print('added joints')

    return  index, all_joints, all_pose

def faiss_idx_np(amass_datas, root_dir,  device='cuda:0'):
    all_pose = []
    print(len(amass_datas))


    print(amass_datas)

    for amass_data in amass_datas:
        ds_dir = os.path.join(root_dir,amass_data)
        seqs = sorted(os.listdir(ds_dir))
        print(amass_data, len(seqs))


        for seq in seqs:
            if not 'npz' in seq:
                continue
            pose_seq = np.load(os.path.join(ds_dir, seq))['pose_body'][:, :63]
            pose_seq = torch.from_numpy(pose_seq.reshape(len(pose_seq), 21, 3))
            pose_seq = axis_angle_to_quaternion(pose_seq).detach().numpy()
            pose_seq, _ = quat_flip(pose_seq)  

            all_pose.extend(pose_seq.reshape(len(pose_seq), 84))
    all_pose = np.array(all_pose)

    # index = faiss.index_factory(84, "Flat",faiss.METRIC_INNER_PRODUCT) (4022919, 84)
    index = faiss.index_factory(84, "Flat")
    index.train(all_pose)
    index.add(all_pose)
    return  index, None, all_pose



def faiss_idx_torch(amass_datas, root_dir):
    """not used"""
    all_pose = []
    for amass_data in amass_datas:
        ds_dir = os.path.join(root_dir,amass_data)
        seqs = sorted(os.listdir(ds_dir))

        for seq in seqs:
            if not 'npz' in seq:
                continue
            # data = np.memmap(os.path.join(ds_dir, seq))
            # tmp+= data.size
            # print('cumulative....', tmp)
            data_val = np.load(os.path.join(ds_dir, seq))['pose_body']
            pose_pt = torch.from_numpy(data_val.reshape(len(data_val), 21, 3))
            # data_val = axis_angle_to_quaternion(pose_pt).detach().numpy()
            data_val = axis_angle_to_quaternion(pose_pt)
            # Todo: add  double cover
            #change data to quaternion

            all_pose.extend(data_val.reshape(len(data_val), 84).unsqueeze(0))
    # print('total size....', tmp)
    print('total data szie......', len(all_pose))

    data_all = torch.cat(all_pose).to(device='cuda')
    res = faiss.StandardGpuResources()
    nlist = 10000   #Todo chnage this number, I think more is faster
    # index = faiss.GpuIndexIVFFlat(res, 84, nlist, faiss.METRIC_L2)
    index = faiss.GpuIndexFlatL2(res, 84)
    index.train(data_all)
    index.add(data_all)

    return  index,data_all
def main(args):

    device ='cuda'
    #read the pose data:
    seq_file = os.path.join(args.raw_data, args.seq_file)
    if not os.path.exists(seq_file):
        print('Missing sequence file.....', seq_file)
        return

    os.makedirs(os.path.join(args.out_dir, args.seq_file.split('/')[0]),exist_ok=True)
    out_dsdir = os.path.join(args.out_dir, args.seq_file.split('/')[0])
    os.makedirs(out_dsdir,exist_ok=True)

    #Create dataloader
    query_data = PoseData(seq_file,mode='query', batch_size=1,num_samples=args.num_samples,runs=args.runs )
    query_data_loader = query_data.get_loader()

    #create distance calcultor
    distance_calculator = getattr(dist_utils, args.metric)
    distance_calculator = distance_calculator(args.batch_size, device)


    amass_datas = sorted(amass_splits['train'])
    # create FAISS search 
    if args.input == 'quat':
        faiss_model, _, all_pose = faiss_idx_np(amass_datas, args.raw_data)
    else:
        # create SMPL 
        num_betas=10
        batch_size = args.num_samples 
        bm_path='/BS/garvita/work/SMPL_data/models'
        bm = smplx.create(bm_path, num_betas=num_betas, batch_size=500).to(device=device)
        betas = torch.zeros((500,10)).to(device=device)
        faiss_model, _, all_pose = faiss_idx_np_jts(amass_datas, args.raw_data, bm, batch_size=500)

    
    print('prepared faiss index.......', len(all_pose))
    batch_size = args.num_samples 
    k_faiss = args.k_faiss
    k_dist = args.k_dist

    all_dists = []
    all_poses = []
    all_nn_poses = []
    idx = 0
    for query_batch in query_data_loader:
        quer_pose_quat = query_batch.get('pose').to(device=device)[0]
        quer_pose_np = quer_pose_quat.reshape(-1, 84).detach().cpu().numpy()
       
        #for every query pose fine knn using faiss and then calculate exact enighbous using custom distance
        distances, neighbors = faiss_model.search(quer_pose_np, k_faiss)
        nn_poses = all_pose[neighbors].reshape(batch_size, k_faiss, 21, 4)

        if args.data_type == 'np':
            nn_poses =  torch.from_numpy(nn_poses).to(device=device)

        print('calculating distance using geodesic distance calculator for {} poses'.format(len(quer_pose_quat)))
        dist, nn_id = distance_calculator.dist_calc(quer_pose_quat,nn_poses, k_faiss, k_dist)

        nn_id = nn_id.detach().cpu().numpy()
        nn_poses = nn_poses.detach().cpu().numpy()
        nn_pose = []
        for idx in range(batch_size):
            nn_pose.append(nn_poses[idx][nn_id[idx]])
        
        all_dists.extend(dist.detach().cpu().numpy())
        all_poses.extend(quer_pose_quat.detach().cpu().numpy())
        all_nn_poses.extend(np.array(nn_pose))
        
    print('done for....{}, pose_shape...{}'.format(args.seq_file, len(all_poses)))
    np.savez(os.path.join(args.out_dir, args.seq_file), dist=np.array(all_dists),nn_pose =np.array(all_nn_poses), pose=np.array(all_poses))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparing pose and distance paired data for training PoseNDF")

    parser.add_argument("-rd", "--raw_data", type=str, default="/BS/humanpose/static00/data/PoseNDF_raw/smpl_h2",
                        help="Path to the resulting image")
    parser.add_argument("-od", "--out_dir", type=str,
                        default="/BS/humanpose/static00/data/PoseNDF_train/smpl_h_flips",
                        help="Path to the resulting image")
    parser.add_argument("-m", "--metric", type=str,
                        default='geo',
                        help="metric used for calculating nn")
    parser.add_argument("-in", "--input", type=str,
                        default='quat',
                        help="calculate nn using difference in joints or quaternions")
    parser.add_argument("-dp", "--data_type", type=str,
                        default='np',
                        help="metric used for calculating nn")
    parser.add_argument("-sf", "--seq_file", type=str,
                        default='BioMotionLab_NTroje/rub001.npz',
                        help="sequence file")
    parser.add_argument("-n", "--num_samples", default=10000, type=int,
                        help="Number of pose sampled from each sequence")
    parser.add_argument("-kr", "--runs", default=10, type=int,

                        help="K nearest neighbour")
    parser.add_argument("-k", "--k_dist", default=5, type=int,

                        help="K nearest neighbour")
    parser.add_argument("-kf", "--k_faiss", default=500, type=int,

                        help="K nearest neighbour")
    parser.add_argument("-bs", "--batch_size", default=128, type=int,
                        help="K nearest neighbour")

    arguments = parser.parse_args()

    if not os.path.exists(os.path.join(arguments.out_dir, arguments.seq_file)):
        main(arguments)
    print('done.....', arguments.seq_file)
