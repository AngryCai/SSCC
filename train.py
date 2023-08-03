import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from modules import HSI_Loader
from torch.utils.data import DataLoader
from torchsummary import summary
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train():
    loss_epoch = 0
    for step, ((x_i, x_j), y_i) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        h_i, h_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(c_i, c_j)
        loss_cross_cor = criterion_cross_cor(c_i, c_j)

        loss = args.loss_tradeoff * loss_instance + loss_cross_cor #+ 0.005 * loss_bce  # loss_instance + #loss_bce + loss_cluster


        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        if step % 50 == 0:
            print(f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cor: {loss_cross_cor.item()}")
                  # f"\t loss_consistency: {loss_bce.item()}")
            # print(f"Step [{step}/{len(data_loader)}]\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = args.dataset_root

    # prepare data
    if args.dataset == "HSI-SaA":
        im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    elif args.dataset == "HSI-InP":
        im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    elif args.dataset == "HSI-InP-2010":
        im_, gt_ = 'Indian_Pines_2010', 'Indian_Pines_2010_gt'
    elif args.dataset == "HSI-SaN":
        im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
    elif args.dataset == "HSI-PaC":
        im_, gt_ = 'Pavia', 'Pavia_gt'
    elif args.dataset == "HSI-Hou":
        im_, gt_ = 'Houston', 'Houston_gt'
    else:
        raise NotImplementedError
    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    dataset = HSI_Loader.HSI_Data(img_path, gt_path, patch_size=(args.image_size, args.image_size), pca=True,
                                  pca_dim=args.in_channel, is_labeled=True,
                                  transform=transform.Transforms(size=args.image_size, s=0.5))
    class_num = dataset.n_classes

    print('Processing %s ' % img_path)
    print(dataset.data_size, dataset.n_classes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    # initialize model
    res = resnet.get_resnet(args.resnet, args.in_channel)
    summary(res, (args.in_channel, args.image_size, args.image_size), device='cpu')
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cross_cor = contrastive_loss.CrossCorrelationLoss(class_num, args.balow_twins_tradeoff, loss_device).to(loss_device)
    loss_history = []
    # train
    save_model(args, model, optimizer, -1)
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print('%s, LR %s ==============================================' % (epoch, lr))
        loss_epoch = train()
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
        loss_history.append(loss_epoch / len(data_loader))
        lr_scheduler.step()
    save_model(args, model, optimizer, args.epochs)
    print(loss_history)
