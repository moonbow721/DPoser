from __future__ import division
from model.posendf import PoseNDF
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn
from model.network.net_utils import gradient
import ipdb
from model.load_data import PoseData
from model.loss_utils import AverageMeter
import shutil

class PoseNDF_trainer(object):

    def __init__(self, opt):

        self.device = opt['train']['device']
        self.enc_name = 'Raw'
        if opt['model']['StrEnc']['use']:
            self.enc_name = opt['model']['StrEnc']['name']
        self.train_dataset = PoseData('train', data_path=opt['data']['data_dir'], amass_dir=opt['data']['amass_dir'],  batch_size=opt['train']['batch_size'], num_workers=opt['train']['num_worker'], num_pts=opt['data']['num_pts'], single= opt['data']['single'], flip=opt['data']['flip'])
        # self.val_dataset = PoseData('train', data_path=opt['data']['data_dir'],  batch_size=opt['train']['batch_size'], num_workers=opt['train']['num_worker'],flip=opt['data']['flip'])
        self.train_dataset  = self.train_dataset.get_loader()
        # self.val_dataset  = self.val_dataset.get_loader()
        # create all the models and dataloader:
        self.learning_rate = opt['train']['optimizer_param']
        self.model = PoseNDF(opt).to(self.device)
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        self.batch_size= opt['train']['batch_size']
        self.ep = 0
        self.val_min = 10000.
        ##initialise the network
        self.init_net(opt)

        if opt['train']['continue_train']:
            self.ep = self.load_checkpoint()
        
        # if opt['train']['inference']:
        #     self.ep = self.load_checkpoint()
        

    
    def init_net(self, opt):
          #geo_weights = np.load(os.path.join(DATA_DIR, 'real_g5_geo_weights.npy'))  todo: do we need this???
        self.d_tol = 0.002
        self.iter_nums = 0 

        #create exp name based on experiment params
        self.loss_weight = {'man_loss':  opt['train']['man_loss'], 'dist': opt['train']['dist'], 'eikonal': opt['train']['eikonal']}
       
        self.exp_name = opt['experiment']['exp_name']
        self.loss = opt['train']['loss_type']

        self.exp_name = '{}_{}_{}_{}__{}_dist{}_eik{}_man{}'.format(self.exp_name,  opt['model']['DFNet']['act'],self.loss,  opt['train']['optimizer_param'], opt['data']['num_pts'],  opt['train']['dist'], opt['train']['eikonal'], opt['train']['man_loss'])
        if opt['data']['flip']:
            self.exp_name = 'flip_{}'.format(self.exp_name)

        self.exp_path = '{}/{}/'.format( opt['experiment']['root_dir'],self.exp_name )
        self.checkpoint_path = self.exp_path + 'checkpoints/'
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary')
        self.loss = opt['train']['loss_type']
        self.n_part = opt['experiment']['num_part']
        self.loss_mse = torch.nn.MSELoss()

        self.batch_size = opt['train']['batch_size']
               

        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()



    def train_model(self, ep=None):

        self.model.train()
        epoch_loss = AverageMeter()
        individual_loss_epoch = {}
        for loss_term in self.loss_weight.keys():
            if self.loss_weight[loss_term]:
                individual_loss_epoch[loss_term] = AverageMeter()

        for i, inputs in enumerate(self.train_dataset):
            self.optimizer.zero_grad()
            _, loss_dict = self.model(inputs['pose'], inputs['dist'], inputs['man_poses'], eikonal=self.loss_weight['eikonal'] )
            loss = 0.0
            for k in loss_dict.keys():
                loss += self.loss_weight[k]*loss_dict[k]
            loss.backward()
            self.optimizer.step()

            epoch_loss.update(loss, self.batch_size)
            for loss_term in self.loss_weight.keys():
                if self.loss_weight[loss_term]:
                    individual_loss_epoch[loss_term].update(loss_dict[loss_term], self.batch_size)

            self.iter_nums +=1
            # logger and summary writer
            for k in loss_dict.keys():
                self.writer.add_scalar("train/Iter_{}".format(k), loss_dict[k].item(), self.iter_nums )
        self.writer.add_scalar("train/epoch", epoch_loss.avg, ep)
        for k in loss_dict.keys():
            self.writer.add_scalar("train/epoch_{}".format(k), individual_loss_epoch[k].avg, ep )
        print( "train/epoch", epoch_loss.avg, ep)

        self.save_checkpoint(ep)
        return loss.item(),epoch_loss.avg
    
    # def inference(self, epoch, eval=True):
    #     self.model.eval()
    #     sum_val_loss = 0

    #     val_data_loader = self.val_dataset
    #     out_path = os.path.join(self.exp_path, 'latest_{}'.format(epoch))
    #     os.makedirs(out_path, exist_ok=True)
    #     for batch in val_data_loader:
    #         loss_dict, output= self.model(batch)
    #         sum_val_loss += loss_dict['data'].item()
    #         print( loss_dict['data'].item())
    #     val_loss =  sum_val_loss /len(val_data_loader)
    #     self.writer.add_scalar("validation_test/epoch", val_loss, epoch)

    #     return val_loss

    # def validate(self, epoch, eval=True):
    #     self.model.eval()
    #     sum_val_loss = 0

    #     val_data_loader = self.val_dataset
    #     for batch in val_data_loader:
    #         loss_dict, _ = self.model(batch)
    #         sum_val_loss += loss_dict['data'].item()
    #     val_loss =  sum_val_loss /len(val_data_loader)
    #     self.writer.add_scalar("validation_vert/epoch", val_loss, epoch)

    #     if val_loss < self.val_min:
    #         self.val_min = val_loss
    #         self.save_checkpoint(epoch)
    #     print('validation vertices loss at {}....{:08f}'.format(epoch,val_loss))
    #     return val_loss



    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_best.tar'

        if not os.path.exists(path):
            torch.save({'epoch':epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path,  _use_new_zipfile_serialization=False)
        else:
            shutil.copyfile(path, path.replace('best', 'previous'))
            torch.save({'epoch':epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, path,  _use_new_zipfile_serialization=False)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        # checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        # checkpoints = np.array(checkpoints, dtype=int)
        # checkpoints = np.sort(checkpoints)
        # path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])
        path = self.checkpoint_path + 'checkpoint_epoch_best.tar'

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loaded checkpoint from: {}'.format(path))

        epoch = checkpoint['epoch']
        return epoch

    
    