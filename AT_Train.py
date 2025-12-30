import torch
import numpy as np
import random
import pdb, os, argparse
from datetime import datetime
import sys
from tqdm import tqdm
from setting.VLdataLoader import get_loader
from setting.utils import clip_gradient, adjust_lr
from network.VLINet import VLINet, TextEncoder
from metrics.SOD_metrics import SODMetrics
from torch.utils.tensorboard import SummaryWriter
from setting.loss_function2 import SaliencyLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--trainset_path", type=str, 
        default="/fuyuxiang/Projects/CLS/Datasets/RGB-DSOD11/RGB-DSOD/RGBD_Train/")
parser.add_argument("--testset_path", type=str, 
        default="/fuyuxiang/Projects/CLS/Datasets/RGB-DSOD11/RGB-DSOD/DUT-RGBD-Test/")
       # RGBD Dataset: [DUT-RGBD-Test, NJUD, NLPR, SIP, STERE]
parser.add_argument("--dataset", type=str, default='RGBD', 
                    help='Name of dataset:[VDT3K]')
parser.add_argument('--model', type=str, default='VLINet', 
                    help='model name:[VLINet]')
parser.add_argument('--epoch', type=int, default=45, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument("--load", type=str, default=None, help="restore from checkpoint")
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument("--n_cpu", type=int, default=8, help="num of workers")
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--save_path', type=str, default='./CHKP/', help='checkpoint save path')
parser.add_argument('--log_dir', type=str, default='./log26/', help='tensorboard log path')
parser.add_argument('--save_ep', type=int, default=5, help='save checkpoint every n epochs')
opt = parser.parse_args()


def validate(opts, model, text_encoder, loader, device, metrics):
    metrics.reset()
    with torch.no_grad():
        for step, (images, depths, texts, labels) in tqdm(enumerate(loader)):
  
            images = images.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            texts_feat = text_encoder(texts).float()
            outputs = model(images, depths, texts_feat)
            sal_map = torch.sigmoid(outputs)
            # 特征维度压缩（适配metrics计算）
            preds = sal_map.squeeze(1)  # (B, 1, H, W) → (B, H, W)
            labels = labels.squeeze(1)  # (B, 1, H, W) → (B, H, W)

            metrics.update(preds, labels)

        score = metrics.get_results()
    return score

if __name__=='__main__':
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    opt.log_dir = opt.log_dir + f'{opt.model}_{opt.dataset}_ep{opt.epoch}_lr{str(opt.lr)}'
    tb_writer = SummaryWriter(opt.log_dir)
    print(f"[Config] {opt}")

    model = eval(opt.model)().to(device)
    text_encoder = TextEncoder("ViT-B/16")
    text_encoder.eval()
    
    if opt.load and os.path.isfile(opt.load):
        checkpoint = torch.load(opt.load, map_location=device)
        # 若 checkpoint 是字典（含model_state_dict），则取value；否则直接加载
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[Checkpoint] Restored from {opt.load}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=opt.lr, weight_decay=1e-5)  # 加权重衰减防过拟合
    metrics = SODMetrics(cuda=True)
    criterion = SaliencyLoss()

    train_loader, train_num = get_loader(
        opt.trainset_path+'train_images/',
        opt.trainset_path+'train_depth/', 
        opt.trainset_path+'train_masks/', 
        opt.trainset_path+'train_text_oct/',  
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=opt.n_cpu
    )
    val_loader, val_num = get_loader(
        opt.testset_path+'test_images/',
        opt.testset_path+'test_depth/', 
        opt.testset_path+'test_masks/', 
        opt.testset_path+'test_text_oct/',
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=opt.n_cpu,
    )
    print(f"[Data] Loaded {train_num} train images, {val_num} val images")
    
    print("Start training...")
    cur_epoch = 0
    best_MAE = 1.0
    for epoch in range(cur_epoch, opt.epoch):
        # 调整学习率
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        model.train()
        running_total_loss = 0.0
        data_loader = tqdm(train_loader, file=sys.stdout)

        for i, (images, depths, texts, gts) in enumerate(data_loader, start=1):
            images = images.to(device)
            depths = depths.to(device)
            gts = gts.to(device)
            
            texts_feat = text_encoder(texts).float()
            outputs = model(images, depths, texts_feat)
            total_loss = criterion(outputs, gts)['total_loss']

            optimizer.zero_grad()
            total_loss.backward()
            clip_gradient(optimizer, opt.clip) 
            optimizer.step()

            running_total_loss += total_loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            data_loader.desc = f"Epoch {epoch+1}/{opt.epoch}, LR: {current_lr:.6f}, Loss: {running_total_loss/i:.4f}"

        print(f"[Val] Epoch {epoch+1} validation...")
        model.eval()
        val_score = validate(opts=opt, model=model, text_encoder=text_encoder, loader=val_loader, device=device, metrics=metrics)
        print(f"[Val] Epoch {epoch+1}, MAE: {val_score['MAE']:.4f}, Sm: {val_score['Sm']:.4f}")

        tags = ["train_loss", "learning_rate", "MAE", "Sm",]
        tb_writer.add_scalar(tags[0], running_total_loss/len(train_loader), epoch)
        tb_writer.add_scalar(tags[1], current_lr, epoch)
        tb_writer.add_scalar(tags[2], val_score["MAE"], epoch)
        tb_writer.add_scalar(tags[3], val_score["Sm"], epoch)

        if val_score['MAE'] < best_MAE:
            best_MAE = val_score['MAE']
            cur_save_path = opt.save_path + f'bst_{opt.model}_{opt.dataset}.pth'
            torch.save(model.state_dict(), cur_save_path)
            print(f"Best Model Saved at {cur_save_path}")

        if (epoch+1) % opt.save_ep == 0:
            torch.save(model.state_dict(), opt.save_path + f'last_{opt.model}_{opt.dataset}.pth')
            if (epoch+1)>=30 and (epoch+1) % 15 == 0:
                torch.save(model.state_dict(), opt.save_path + f'{opt.model}_{opt.dataset}_{epoch+1}e.pth')
            print(f"Model Saved at {opt.save_path}")

    tb_writer.close()
    print("Training completed!")
