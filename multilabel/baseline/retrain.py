# import sys
# sys.path.append('/opt/ml/finalproject')

#dependency
from multilabel.baseline.metrics import top_k_labels, get_confusion_matrix, get_metrics_from_matrix
from multilabel.baseline.dataset import RetrainDataset
from multilabel.baseline.transform import val_transform

#else
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
from tqdm import tqdm


def retrain(model: nn.Module, img_dirs: list, epoch: int, save_path: str, batch_size: str = 1):
    retrain_dataset = RetrainDataset(img_dirs, transform = val_transform)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print("retrain start")
    print("pytorch version: {}".format(torch.__version__))
    print("GPU 사용 가능 여부: {}".format(torch.cuda.is_available()))

    loader = DataLoader(
        dataset = retrain_dataset,
        batch_size = batch_size,
        num_workers = 1,
        shuffle = False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    #다양한 옵션으로 실험할 것 아니기에 일부 설정 hardcoding
    #yaml에서 받아오도록 해도 됨
    optimizer = optim.Adam(
            model.parameters(),
            lr=float(1e-2),
            weight_decay=float(1e-5),
            amsgrad=False
    )
    criterion = nn.CrossEntropyLoss()
    
    #wandb visualizing, eval 검증 제외
    #multihead model(jiwoo branch) 기준
    step = 0
    model.to(device)
    model.train()
    for epoch in range(epoch):
        train_emr = []
        train_confusion_matrix = np.zeros((38, 4))
        for (images, labels) in tqdm(loader, desc = f'train/epoch {epoch}', leave = False, total=len(loader)):
            images = images.to(device)
            
            optimizer.zero_grad()
            outs, cls_outs = model(images)

            loss = model.get_loss(
                outs, 
                cls_outs,
                labels, 
                criterion
            )

            preds = top_k_labels(outs, cls_outs, identity = False)
            
            loss.backward()
            optimizer.step()
            
            # EMR/loss
            images = images.detach().cpu()
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            matrix = get_confusion_matrix(preds, labels)
            train_confusion_matrix += np.array(matrix)
            train_emr.append(np.mean((preds == labels).min(axis = 1)))

            print(f'train EMR: {np.mean(train_emr)}')
            print(f'train loss: {round(loss.item(), 4)}')

            step += 1
        
        _, metrics = get_metrics_from_matrix(train_confusion_matrix)
        print(f'Train mAR: {metrics[0]}')
        print(f'Train mAP: {metrics[1]}')
        print(f'Train mAP: {metrics[2]}')

    torch.save(model.state_dict(), save_path)
    print(f'save complete at: {save_path}')


#testcode
if __name__ == '__main__':
    from model import multihead
    model = multihead(38, 0, 'cuda')
    retrain(model, ['/opt/ml/tmp/img/0001[0,1,2].png', '/opt/ml/tmp/img/0002[2,5,20].png'], 3, '/opt/ml/tmp/weight.pth')