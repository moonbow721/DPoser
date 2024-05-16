'''
Initializes the training and validating procedures in a very common way: 
in a separate file PROCEDURE.py user specifies only experiment-specific 
training and validating procedures (with the capability of custom informing per epoch). 
Proposed wrappers do the procedure and also ensure meaningful logging, 
independent from the experiment.
'''

def EPOCH_INFO(trainer):
    return ''

def TRAIN(trainer):
    return

def VALID(trainer):
    return float('inf')


def init_networks_for_training(trainer):
    for model in trainer.models.values():
        model.train()

def init_networks_for_validation(trainer):
    for model in trainer.models.values():
        model.eval()

def train_with_logging(trainer, train):
    trainer.logger.info('=> Training...')
    init_networks_for_training(trainer)

    train(trainer) ### custom sampling with inference and optimization

    for loss_meter in trainer.meters.train.values():
        loss_meter.epochends()

def valid_with_logging(trainer, valid):
    trainer.logger.info('=> Validating...')
    init_networks_for_validation(trainer)

    perf_indicator = valid(trainer) ### custom sampling with inference

    for loss_meter in trainer.meters.valid.values():
        loss_meter.epochends()

    return perf_indicator


def one_epoch(train=TRAIN, valid=VALID, epoch_info=EPOCH_INFO):
    ''' Returns "one_epoch" procedure with specified "train"/"valid"/"epoch_info" functions specified.
    No need to keep these functions separately in the trainer. 
    Assume that only this template of "do_one_epoch" will be used, train,valid,epoch_info 
    must be able to cover all variety of training scenarios.
    '''

    def do_one_epoch(trainer):
        trainer.logger.info(f'=> Epoch [{trainer.cur_epoch}] starts...')

        ### show valuable info if necessary
        epoch_info(trainer)

        ### if needed, do additional validation first:
        if trainer.cur_epoch == 1 and 'valid_first' in trainer.cfg and trainer.cfg.valid_first:
            _ = valid_with_logging(trainer, valid)

        ### do training for one epoch
        train_with_logging(trainer, train)

        ### do validation for one epoch
        trainer.perf_indicator = valid_with_logging(trainer, valid)
        best = trainer.perf_indicator < trainer.perf_indicator_best
        if best:
            trainer.perf_indicator_best = trainer.perf_indicator

        ### save the checkpoint
        trainer.save_state(best=best)

        if trainer.sched.do_scheduling:  
            ### update scheduler
            trainer.sched.step(trainer.perf_indicator)

            ### log in case lr has been changed
            scheds = trainer.sched.sched
            new_lrs = trainer.sched.new_lrs

            for model_name, sched in scheds.items():
                new_lr = new_lrs[model_name]
                if new_lr is not None:
                    trainer.logger.info(f'=> "{model_name}" learning rate reduced to {new_lr:.2e}.')

            ### finish the training in case all lrs got minimum value
            if sum(trainer.sched.sched_finished.values()) == len(trainer.sched.sched_finished):
                trainer.logger.info(f'\nTraining of "{trainer.cfg.EXP_NAME}" experiment is finished. \n')
                exit()

        trainer.logger.info('=> ...')

    return do_one_epoch


def get_procedures_by_config(cfg):
    ''' This function fully comprises the whole procedure.
    It is assumed that file with the procedure <proc_name>.py lies in the folder "procedures".
    <proc_name.py> file must contain at least two functions, 
    describing how one epoch's training and validation must look like.  
    '''
    assert cfg.PROCEDURE != '', 'procedure name must be specified! Check the <exp>.yaml file'
    exec(f'from .procedures import {cfg.PROCEDURE} as proc')

    directory = dir(eval('proc'))
    train = eval('proc.train') if 'train' in directory else TRAIN
    valid = eval('proc.valid') if 'valid' in directory else VALID
    epoch_info = eval('proc.epoch_info') if 'epoch_info' in directory else EPOCH_INFO

    out_proc = one_epoch(train=train, valid=valid, epoch_info=epoch_info)
    return out_proc


def status_msg(trainer, batch_idx, dl_len, lossmeter, total_time):
    '''
    Provides a verbose status of training, what epoch is, what batch is, what current metrics value is
    '''
    if batch_idx % trainer.print_freq == 0 or batch_idx == dl_len or batch_idx == 1:
        # print every first, last and chosen ("print_freq") batches
        msg = (
          f'Epoch: [{trainer.cur_epoch}][{batch_idx:4}/{dl_len:4}]\t'
          f'Total {total_time:5.1f}s \t'
          f'{lossmeter.cur_val:12.5e}, avg {lossmeter.cur_avg:12.5e}'
              )
        trainer.logger.info(msg)






