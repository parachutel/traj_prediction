import torch
import torch.optim as optim
import functools

from args import args

def exp_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    rate = torch.tensor(anneal_kws['rate'], device=device)
    return lambda step: finish - (finish - start) * torch.pow(
        rate, torch.tensor(step, dtype=torch.float, device=device))


def sigmoid_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    center_step = torch.tensor(anneal_kws['center_step'], device=device, dtype=torch.float)
    steps_lo_to_hi = torch.tensor(anneal_kws['steps_lo_to_hi'], device=device, dtype=torch.float)
    return lambda step: start + (finish - start) * torch.sigmoid(
        (torch.tensor(float(step), device=device) - center_step) * (1. / steps_lo_to_hi))


class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def create_scheduler(model, var_name, annealer, annealer_kws, 
                     creation_condition=True):
    value_scheduler = None
    dummy_optimizer = None
    if creation_condition:
        annealer_kws['device'] = model.device
        value_annealer = annealer(annealer_kws)
    
        # This is the value that we'll update on each call of
        # step_annealers().
        rsetattr(model, var_name, torch.tensor(value_annealer(0), device=model.device))
    
        dummy_optimizer = optim.Optimizer([rgetattr(model, var_name)], 
            {'lr': torch.tensor(value_annealer(0), device=model.device)})
        
        value_scheduler = CustomLR(dummy_optimizer, 
                                   value_annealer)

    model.dummy_optimizers.append(dummy_optimizer)
    model.schedulers.append(value_scheduler)
    model.annealed_var_names.append(var_name)


def setup_hyperparams_annealing(model):
    create_scheduler(model,
        'kl_weight',
        annealer=sigmoid_anneal, 
        annealer_kws={
           'start': args.kl_weight_start,
           'finish': args.kl_weight,
           'center_step': args.kl_crossover,
           'steps_lo_to_hi': args.kl_crossover / args.kl_sigmoid_divisor
        },
        creation_condition=(abs(args.alpha - 1.0) < 1e-3 and not args.use_iwae)
    )
    create_scheduler(model,
        'decoder.decoding_sample_model_prob',
        annealer=sigmoid_anneal, 
        annealer_kws={
           'start': args.dec_sample_model_prob_start,
           'finish': args.dec_sample_model_prob_final,
           'center_step': args.dec_sample_model_prob_crossover,
           'steps_lo_to_hi': args.dec_sample_model_prob_crossover / args.dec_sample_model_prob_divisor
        },
        creation_condition=args.sample_model_during_dec
    )
    create_scheduler(model,
        'encoder.latent.temp',
        annealer=exp_anneal, 
        annealer_kws={
            'start': args.temp_init,
            'finish': args.temp_final,
            'rate': args.temp_decay_rate
        }
    )
    create_scheduler(model,
        'encoder.latent.z_logit_clip',
        annealer=sigmoid_anneal,
        annealer_kws={
           'start': args.z_logit_clip_start,
           'finish': args.z_logit_clip_final,
           'center_step': args.z_logit_clip_crossover,
           'steps_lo_to_hi': args.z_logit_clip_crossover / args.z_logit_clip_divisor
        },
        creation_condition=args.use_z_logit_clipping
    )

def step_annealers(model, tbx, step):
    for var_name, scheduler, dummy_optimizer \
        in zip(model.annealed_var_names, model.schedulers, model.dummy_optimizers):
        if scheduler is not None:
            scheduler.step()
            annealed_var = dummy_optimizer.param_groups[0]['lr']
            rsetattr(model, var_name, annealed_var)
            # print('hyper_params/' + var_name, annealed_var.item())
            tbx.add_scalar('hyper_params/' + var_name, annealed_var.item(), step)