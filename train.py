import math
import numpy as np
import random
from json import dumps
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched

from predictor.predictor import Predictor
from model_utils.annealer import step_annealers

from data_utils.highd_dataset import build_highd_data_loader
import util
from args import args

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    # device, args.gpu_ids = 'cpu', None
    log.info(f'Using device {device}...')
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info('Building model...')
    model = Predictor(state_dim=args.state_dim,
                      rel_state_dim=args.state_dim,
                      pred_dim=args.pred_dim,
                      edge_type_dim=args.n_edge_types,
                      nhe_hidden_size=args.nhe_hidden_size,
                      ehe_hidden_size=args.ehe_hidden_size,
                      nfe_hidden_size=args.nfe_hidden_size,
                      decoder_hidden_size=args.decoder_hidden_size,
                      gmm_components=args.gmm_components,
                      log_sigma_min=args.log_sigma_min,
                      log_sigma_max=args.log_sigma_max,
                      log_p_yt_xz_max=args.log_p_yt_xz_max,
                      kl_weight=args.kl_weight,
                      device=device)
    
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    parameters = filter(lambda param: param.requires_grad, 
                        model.parameters())

    # Optimizer
    optimizer = optim.Adam(lr=args.lr, betas=(0.9, 0.999), eps=1e-7, 
                           weight_decay=5e-8, params=parameters)
    # LR scheduler
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.learning_decay_rate)
    
    # Get data loader
    log.info('Building dataset...')
    n_total_data_files = len(args.data_list)
    n_train_data_files = int(0.75 * n_total_data_files)
    random.shuffle(args.data_list)
    train_data_list = args.data_list[:n_train_data_files]
    dev_data_list = args.data_list[n_train_data_files:]
    # Dataset is on CPU first to save VRAM
    train_loader = build_highd_data_loader(train_data_list, args.train_batch_size)
    dev_loader = build_highd_data_loader(dev_data_list, args.eval_batch_size)
    log.info(f'Training set size = {len(train_loader.dataset)}')
    log.info(f'Dev set size = {len(dev_loader.dataset)}')


    # Train
    log.info('Training...')
    epochs_till_eval = args.eval_epochs
    epoch = 0
    step = 0
    while epoch != args.num_epochs:
        epoch += 1

        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            # One epoch:
            for input_seq, _, input_edge_types, pred_seq in train_loader:
                # Move to GPU if needed
                input_seq = input_seq.to(device)
                input_edge_types = input_edge_types.to(device)
                pred_seq = pred_seq.to(device)

                batch_size = input_seq.size(0)
                optimizer.zero_grad()

                # Forward
                loss = model.get_training_loss(input_seq, input_edge_types, pred_seq)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                
                step += batch_size
                # Step hyperparam schedulers
                lr_scheduler.step()
                step_annealers(model, tbx, step)

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         Loss=loss_val)
                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

        # End epoch
        epochs_till_eval -= 1
                
        if epoch == 1 or epochs_till_eval <= 0:
            epochs_till_eval = args.eval_epochs
            # Evaluate and save checkpoint
            log.info(f'Evaluating at epoch {epoch}...')
            results = evaluate(model, dev_loader, device)
            saver.save(epoch, model, results[args.metric_name], device)
            # Log to console
            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
            log.info(f'Dev {results_str}')
            # Log to TB
            tbx.add_scalar('eval/nll_q_is', results['nll_q_is'], step)
            tbx.add_scalar('eval/nll_p', results['nll_p'], step)
            tbx.add_scalar('eval/nll_exact', results['nll_exact'], step)


def evaluate(model, data_loader, device):
    model.eval()
    nll_q_is_meter = util.AverageMeter()
    nll_p_meter = util.AverageMeter()
    nll_exact_meter = util.AverageMeter()

    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for input_seq, _, input_edge_types, pred_seq in data_loader:
            # Move to GPU if needed
            input_seq = input_seq.to(device)
            input_edge_types = input_edge_types.to(device)
            pred_seq = pred_seq.to(device)

            batch_size = input_seq.size(0)
            # Forward
            nll_q_is, nll_p, nll_exact = \
                model.get_eval_loss(input_seq, input_edge_types, pred_seq)

            nll_q_is_meter.update(nll_q_is.item(), batch_size)
            nll_p_meter.update(nll_p.item(), batch_size)
            nll_exact_meter.update(nll_exact.item(), batch_size)
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_exact.item())

    model.train()
    results_list = [('nll_q_is', nll_q_is_meter.avg),
                    ('nll_p', nll_p_meter.avg),
                    ('nll_exact', nll_exact_meter.avg)]
    results = dict(results_list)
    return results


if __name__ == '__main__':
    main(args)
