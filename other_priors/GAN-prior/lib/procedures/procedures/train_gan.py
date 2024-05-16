from time import time
from ..procedures_common import status_msg
from os.path import join as joinpath

from lib.functional.sampling import generate_random_input as _generate_random_input
from lib.models.rodrigues_utils import batch_rodrigues
from lib.plot_utils import plot_pose


def train(trainer):
    absolute_start = time() 

    generate_random_input = lambda batch_size : _generate_random_input(batch_size, trainer.cfg.LATENT_SPACE_DIM, trainer.cfg.LATENT_SPACE_TYPE)

    dl_len = len(trainer.dataload.train)
    for batch_idx, sample in enumerate(trainer.dataload.train, start=1):
        pose_gt = sample['pose'].to(device=trainer.device0, non_blocking=True)
        batch_size = pose_gt.size(0)
        matrot_real = batch_rodrigues(pose_gt.contiguous().view(-1,3))
        matrot_real = matrot_real.view(batch_size, -1, 9)

        #########################
        # Generator step
        all_gen_losses = []
        for i in range(trainer.cfg.NUM_GEN_STEPS):
            trainer.optim.opts.generator.zero_grad()
            
            # gen loss -> pull d_fake to 1 with frozen Discr
            z = generate_random_input(batch_size).to(device=trainer.device0)
            # TODO decide whether we need regularization and wish to operate on 'matrot' or 'aa'
            matrot_fake = trainer.models.generator(z, output_type='matrot')
            d_fake = trainer.models.discriminator(matrot_fake, input_type='matrot')
            generator_loss = trainer.losses.dis_loss(d_fake, label=1.)
            generator_loss.backward()
            trainer.optim.opts.generator.step()
            all_gen_losses.append(generator_loss)
        generator_loss = sum(all_gen_losses) / len(all_gen_losses)
        
        trainer.meters.train.gen_loss.update(generator_loss.item(), n=batch_size)
        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.gen_loss, total_time)

        #########################
        # Discriminator step
        all_dis_losses = []
        for i in range(trainer.cfg.NUM_DIS_STEPS):
            trainer.optim.opts.discriminator.zero_grad()

            # discr fake loss -> pull d_fake to 0 with frozen Generator
            d_fake = trainer.models.discriminator(matrot_fake.detach(), input_type='matrot')
            fake_disc_loss = trainer.losses.dis_loss(d_fake, label=0.)
            
            # discr fake loss -> pull d_real to 1
            d_real = trainer.models.discriminator(matrot_real.detach(), input_type='matrot')
            real_disc_loss = trainer.losses.dis_loss(d_real, label=1.)
            
            discriminator_loss = fake_disc_loss + real_disc_loss
            discriminator_loss.backward()
            trainer.optim.opts.discriminator.step()
            all_dis_losses.append(discriminator_loss)
        discriminator_loss = sum(all_dis_losses) / len(all_dis_losses)
        
        trainer.meters.train.dis_loss.update(discriminator_loss.item(), n=batch_size)
        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.dis_loss, total_time)

        if 'SAVEALLITERS' in trainer.cfg and trainer.cfg.SAVEALLITERS and batch_idx != dl_len:
            trainer.meters.train.gen_loss.epochends()
            trainer.meters.train.dis_loss.epochends()



def valid(trainer):

    ### if "valid_first" then epoch is "0"
    if trainer.cur_epoch == 1 and 'valid_first' in trainer.cfg and trainer.cfg.valid_first:
        epoch = 0
        trainer.cfg.valid_first = False
    else:
        epoch = trainer.cur_epoch

    absolute_start = time()
    ############################################################
    # track discr distribution
    # discr output is the dictionary
    ############################################################
    trainer.logger.info(f'Track discriminator predictions...')

    num_features = 10_000

    d_real, x_real_pose = plot_pose.get_d_real(
        trainer.models.discriminator, 
        trainer.dataload.train,
        num_features=num_features
        )
    
    num_features = len(x_real_pose)

    d_fake, x_fake_pose = plot_pose.get_d_fake(
        trainer.models.generator, 
        trainer.models.discriminator, 
        num_features=num_features,
        batch_size=trainer.cfg.DATALOAD.train.PARAMS.batch_size,
        latent_space_dim=trainer.cfg.LATENT_SPACE_DIM,
        latent_space_type=trainer.cfg.LATENT_SPACE_TYPE
        )
    savepath = joinpath(trainer.final_output_dir, 'discr_hists')
    plot_pose.plot_d_predictions(d_real, d_fake, savepath, add_name=f'{epoch:04d}')
    total_time = time() - absolute_start
    msg = (f'=> Epoch [{epoch}][Discr] Total {total_time:5.1f}s \t')
    trainer.logger.info(msg)
    absolute_start = time()
    ############################################################
    # T-SNE visualization
    ############################################################
    trainer.logger.info(f'Visualize T-SNE...')

    num_features = min(num_features, 1_000 if epoch % 10 != 0 else 3_000)

    savepath = joinpath(trainer.final_output_dir, 'tsne_plots')
    plot_pose.plot_tsne(
            reals=x_real_pose, fakes=x_fake_pose,
            savepath=savepath, num_features=num_features, add_name=f'{epoch:04d}'
        )

    total_time = time() - absolute_start
    msg = (f'=> Epoch [{epoch}][t-SNE] Total {total_time:5.1f}s \t')
    trainer.logger.info(msg)
    ############################################################
    # Poses visualization
    ############################################################
    absolute_start = time()

    trainer.logger.info(f'Visualize poses...')
    savepath = joinpath(trainer.final_output_dir, 'poses')
    plot_pose.plot_poses(trainer, x_fake_pose, savepath, add_name=f'{epoch:04d}')
    total_time = time() - absolute_start
    msg = (f'=> Epoch [{epoch}][Poses] Total {total_time:5.1f}s \t')
    trainer.logger.info(msg)

    perf_indicator = float('inf')
    return perf_indicator