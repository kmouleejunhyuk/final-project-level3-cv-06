import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_opt_sche(config_train, model):
    """
    define optimizer & scheduler
    """
    param_groups = model.parameters()

    if config_train['optimizer'] == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=config_train['lr'],
            weight_decay=config_train['weight_decay'],
            momentum=config_train['momentum'],
            nesterov=False,
        )

    elif config_train['optimizer'] == "adam":
        # print(config_train['amsgrad'], type(config_train['lr']))
        optimizer = optim.Adam(
            param_groups,
            lr=float(config_train['lr']),
            weight_decay=float(config_train['weight_decay']),
            amsgrad=config_train['amsgrad']
        )
    # elif args.optimizer == "radam":
    #     optimizer = RAdam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    
    elif config_train['optimizer'] == 'AdamW':
        lr_mult = config_train['batch_size'] / 256
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
        optimizer = optim.AdamW(
            param_dicts,
            lr_mult * config_train['lr'],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=config_train['weight_decay']
        )

    elif config_train['optimizer'] == 'Adam_twd':
        lr_mult = config_train['batch_size'] / 256
        parameters = add_weight_decay(model, 0.1)   #origin yaml
        optimizer = optim.Adam(
            parameters,
            lr_mult * config_train['lr'],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )

    else:
        raise ValueError("Not a valid optimizer")


    poly_schd = lambda epoch: math.pow(1 - epoch / config_train['epochs'], config_train['poly_exp'])

    if config_train['scheduler'] == "lambda":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schd)
    elif config_train['scheduler'] == "cosineanneal":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config_train['T_max'], eta_min=config_train['eta_min']
        )
    elif config_train['scheduler'] == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=config_train['step_size'], gamma=config_train['gamma']
        )
    elif config_train['scheduler'] == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30])

    return optimizer, scheduler
