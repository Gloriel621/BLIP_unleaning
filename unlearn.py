import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import copy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.utils.clip_grad as clip_grad
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from models.blip_pretrain import blip_pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_split_loaders

def discrepancy_loss(original_repr, unlearned_repr):
    return F.mse_loss(original_repr, unlearned_repr)

def unlearn(model, original_model, data_loader, optimizer, epoch, device, config, lambda1, lambda2, lambda3):
    model.train()
    original_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('discrepancy_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('lambda1', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('lambda2', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('lambda3', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Unlearn Epoch: [{}]'.format(epoch)
    print_freq = 10

    data_loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        try:
            image, caption = batch

            if epoch == 0:
                warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])

            optimizer.zero_grad()

            image = image.to(device, non_blocking=True)

            alpha = adjust_alpha(epoch, i, len(data_loader), 'unlearn', config)

            loss_ita, loss_itm, loss_lm, image_repr, text_repr = model(image, caption, alpha=alpha)
            # negative gradient with adaptive lambdas
            loss = -(lambda1 * loss_ita + lambda2 * loss_itm + lambda3 * loss_lm)

            # Get original model representations
            with torch.no_grad():
                _, _, _, original_image_repr, original_text_repr = original_model(image, caption, alpha=alpha)

            # Add custom discrepancy loss
            image_disc_loss = discrepancy_loss(original_image_repr, image_repr)
            text_disc_loss = discrepancy_loss(original_text_repr, text_repr)
            disc_loss = image_disc_loss + text_disc_loss

            disc_lambda = config.get('disc_lambda', 1)
            loss += disc_lambda * disc_loss

            loss.backward()

            clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            metric_logger.update(loss_ita=loss_ita.item())
            metric_logger.update(loss_itm=loss_itm.item())
            metric_logger.update(loss_lm=loss_lm.item())
            metric_logger.update(discrepancy_loss=disc_loss.item())
            metric_logger.update(lambda1=lambda1.item())
            metric_logger.update(lambda2=lambda2.item())
            metric_logger.update(lambda3=lambda3.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        except Exception as e:
            print(f"Caught an exception in DataLoader at iteration {i}: {str(e)}")
            continue

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def retrain(model, data_loader, optimizer, epoch, device, config):

    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))    
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Retrain Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    data_loader.sampler.set_epoch(epoch)

    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
            
        optimizer.zero_grad()
        
        image = image.to(device,non_blocking=True)
        
        # ramp up alpha in the first 2 epochs
        alpha = adjust_alpha(epoch, i, len(data_loader), 'retrain', config)

        loss_ita, loss_itm, loss_lm, _, __ = model(image, caption, alpha = alpha)  
        loss = loss_ita + loss_itm + loss_lm  

        loss.backward()

        clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()    

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

def evaluate(model, data_loader, device, alpha=0.0):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    with torch.no_grad():
        for batch_number, (images, captions) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            # Using the alpha value directly in the model's inference call
            loss_ita, loss_itm, loss_lm, _, __ = model(images, captions, alpha=alpha)

            metric_logger.update(loss_ita=loss_ita.item())
            metric_logger.update(loss_itm=loss_itm.item())
            metric_logger.update(loss_lm=loss_lm.item())
            metric_logger.update(lr=0)

    print("Final Evaluation Results:")
    
    for key, meter in metric_logger.meters.items():
        print(f"{key}: {meter.global_avg:.4f}")

    return {k: f"{meter.global_avg:.4f}" for k, meter in metric_logger.meters.items()}

def adjust_alpha(epoch, i, total_batches, phase='unlearn', config={}):
    base_alpha = config['alpha']
    return base_alpha * min(1, (epoch * total_batches + i) / (2 * total_batches))

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")

    # create forget and retain loaders.
    datasets = [create_dataset('pretrain', config, min_scale=0.2)]
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    loaders = create_split_loaders(datasets, num_tasks, global_rank, [config['batch_size']], [4], split_ratio=0.005)

    forget_loader, retain_loader = loaders[0] 

    #### Model #### 
    print("Loading model")
    # load the pretrained model from checkpoint.
    if not args.evaluate:
        model = blip_pretrain(pretrained=config['pretrained'], image_size=config['image_size'],
                              vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                              vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])
        original_model = copy.deepcopy(model)
        model = model.to(device)
        original_model = original_model.to(device)

        optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': base_lr},
        {'params': [lambda1, lambda2, lambda3], 'lr': lambda_lr}
        ], weight_decay=config['weight_decay'])

    # Initialize lambda values as learnable parameters
    lambda1 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
    lambda2 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
    lambda3 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

    # Define different learning rates for model parameters and lambda parameters
    base_lr = config['init_lr']
    lambda_lr = 1  # Set a different learning rate for lambda parameters


    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict) 

    if args.evaluate:
        # Evaluation mode
        if args.unlearned:
            baseline_model = blip_pretrain(pretrained=config['pretrained'], image_size=config['image_size'],
                              vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                              vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])
            baseline_model = baseline_model.to(device)
            unlearned_model = blip_pretrain(pretrained=config['unlearned'], image_size=config['image_size'],
                                             vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                                             vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])
            unlearned_model = unlearned_model.to(device)   
  
        print("Evaluating the unlearned model...")
        print("Baseline model + forget set")
        evaluate(baseline_model, forget_loader, device)
        print("Unlearned model + forget set")
        evaluate(unlearned_model, forget_loader, device) 

        # use 5% of retain set to evaluate
        retain_size = len(retain_loader.dataset)
        subset_size = int(0.05 * retain_size)
        sampled_indices = np.random.choice(retain_size, size=subset_size, replace=False)
        sampled_retain_set = Subset(retain_loader.dataset, sampled_indices)

        sampled_retain_loader = DataLoader(
            sampled_retain_set,
            batch_size=config['batch_size'],
            num_workers=4,
            sampler=DistributedSampler(sampled_retain_set),
            drop_last=True
        )

        print("Unlearned model + retain set 5%")
        evaluate(unlearned_model, sampled_retain_loader, device)
  
         
    else:
        # Unlearn mode. 
        start_epoch = 0
        if args.checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1                
            print('resume checkpoint from %s'%args.checkpoint)    
    
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module    
        
        print("Start Unlearning")
        start_time = time.time()    
        for epoch in range(start_epoch, config['max_epoch']):
            
            step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
                    
            unlearn_stats = unlearn(model, original_model, forget_loader, optimizer, epoch, device, config, lambda1, lambda2, lambda3)

            subset_size = len(forget_loader.dataset)
            sampled_indices = np.random.choice(len(retain_loader.dataset), size=subset_size * 2, replace=False)
            sampled_retain_set = Subset(retain_loader.dataset, sampled_indices)

            sampled_retain_loader = DataLoader(
                sampled_retain_set,
                batch_size=config['batch_size'],
                num_workers=4,
                sampler=DistributedSampler(sampled_retain_set),
                drop_last=True
            )
            
            retrain_stats = retrain(model, sampled_retain_loader, optimizer, epoch, device, config)

            # for logging.
            if utils.is_main_process():  
                log_stats = {
                    **{f'unlearn_{k}': v for k, v in unlearn_stats.items()},
                    **{f'retrain_{k}': v for k, v in retrain_stats.items()},
                    'epoch': epoch,
                    'lambda1': lambda1.item(),
                    'lambda2': lambda2.item(),
                    'lambda3': lambda3.item(),
                }                 
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                current_checkpoint_path = os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch)
                previous_checkpoint_path = os.path.join(args.output_dir, 'checkpoint_%02d.pth' % (epoch - 1))

                torch.save(save_obj, current_checkpoint_path)
                if os.path.exists(previous_checkpoint_path):
                    os.remove(previous_checkpoint_path)

                with open(os.path.join(args.output_dir, "unlearn_log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()        
                    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Total Unlearn time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/unlearn_pretrain.yaml')
    parser.add_argument('--output_dir', default='output/unlearn')  
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--evaluate', action='store_true')  
    parser.add_argument('--unlearned', default=True)  
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
