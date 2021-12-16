import math

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


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
    else:
        raise ValueError("Not a valid optimizer")

    def poly_schd(epoch):
        return math.pow(1 - epoch / config_train['epochs'], config_train['poly_exp'])

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
    else:
        scheduler = None

    return optimizer, scheduler
