import torch
import torch.nn as nn
from torch import optim
import numpy as np

import os
import os.path as osp
import json
from optparse import OptionParser
from prune_tools import Pruner
from tqdm import tqdm
from finetune import finetune
from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, get_logger, get_save_dir


def get_args():
    parser = OptionParser()
    parser.add_option('-n', '--name', dest='name',
                      default="initial", help='run name')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=2,
                      type='int', help='batch size')
    parser.add_option('-t', '--taylor_batches', dest='taylor_batches', default=500,
                      type='int', help='number of mini-batches used to calculate Taylor criterion')
    parser.add_option('-p', '--prune_channels', dest='prune_channels', default=300,
                      type='int', help='number of channels to remove')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load',
                      default="MODEL.pth", help='load file model')
    parser.add_option('-c', '--channel_txt', dest='channel_txt',
                      default="model_channels.txt", help='load channel txt')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-r', '--lr', dest='lr', type='float',
                      default=0.1, help='learning rate for finetuning')
    parser.add_option('-i', '--iters', dest='iters', type='int',
                      default=1500, help='number of mini-batches for fine-tuning')
    parser.add_option('-e', '--epochs', dest='epochs', type='int',
                      default=None, help='number of epochs for final finetuning')
    parser.add_option('-f', '--flops', dest='flops_reg', type='float',
                      default=.001, help='FLOPS regularization strength')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    # Book-keeping & paths
    args = get_args()

    dir_img = 'data/train/train/'
    dir_mask = 'data/train_masks/train_masks/'

    valid_img = 'data/valid/valid/'
    valid_mask = 'data/valid_masks/valid_masks/'

    dir_checkpoint = 'save/'
    splitfile = "data/trainval.json"

    runname = args.name
    save_path = os.path.join(dir_checkpoint, runname)
    save_dir = get_save_dir(save_path, runname, training=False)  # unique save dir
    log = get_logger(save_dir, runname)  # logger
    log.info('Args: {}'.format(json.dumps({"batch_size": args.batch_size,
                                           "taylor_batches": args.taylor_batches,
                                           "prune_channels": args.prune_channels,
                                           "gpu": args.gpu,
                                           "load": args.load,
                                           "channel_txt": args.channel_txt,
                                           "scale": args.scale,
                                           "lr": args.lr,
                                           "iters": args.iters,
                                           "epochs": args.epochs,
                                           "flops_reg": args.flops_reg},
                                          indent=4, sort_keys=True)))

    # Dataset


    images = os.listdir(dir_img)
    images = [dir_img + img for img in images]
    masks = os.listdir(dir_mask)
    masks = [dir_mask + mask for mask in masks]

    valid_images = os.listdir(valid_img)
    valid_images = [valid_img + img for img in valid_images]
    valid_masks = os.listdir(valid_mask)
    valid_masks = [valid_mask + mask for mask in valid_masks]

    train = get_imgs_and_masks(images, masks, 0.5)
    valid = get_imgs_and_masks(valid_images, valid_masks, 0.5)


    print(f"scale:{args.scale}")
    # Model Initialization
    print(args.channel_txt)
    net = UNet(n_channels=3, n_classes=1, f_channels=args.channel_txt)
    log.info("Built model using {}...".format(args.channel_txt))
    if args.gpu:
        net.cuda()
    if args.load:
        net.load_state_dict(torch.load(args.load))
        log.info('Loading checkpoint from {}...'.format(args.load))
    print("flops",args.flops_reg)
    pruner = Pruner(net, args.flops_reg)  # Pruning handler
    criterion = nn.BCELoss()

    # Ranking on the train dataset
    log.info("Evaluating Taylor criterion for %i mini-batches" % args.taylor_batches)

    print(args.taylor_batches,args.batch_size)
    print(f"channels:{args.prune_channels}")
    with tqdm(total=1) as progress_bar:
        for i, b in enumerate(batch(train,4)):

            net.zero_grad()  # Zero gradients. DO NOT ACCUMULATE

            # Data & Label
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            if args.gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # Forward pass
            masks_pred = net(imgs).squeeze()

            # Backward pass
            loss = criterion(masks_pred, true_masks)
            loss.backward()

            # Compute Taylor rank
            if i == 0:
                log.info("FLOPs before pruning: \n{}".format(pruner.calc_flops()))
            pruner.compute_rank()

            # Tracking progress
            progress_bar.update(args.batch_size)
            if i == args.taylor_batches:  # Stop evaluating after sufficient mini-batches
                log.info("Finished computing Taylor criterion")
                break

    # Prune & save
    pruner.pruning(args.prune_channels)
    log.info('Completed Pruning of %i channels' % args.prune_channels)

    save_file = osp.join(save_dir, "Pruned.pth")
    torch.save(net.state_dict(), save_file)
    log.info('Saving pruned to {}...'.format(save_file))

    save_txt = osp.join(save_dir, "pruned_channels.txt")
    pruner.channel_save(save_txt)
    log.info('Pruned channels to {}...'.format(save_txt))

    del net, pruner
    net = UNet(n_channels=3, n_classes=1, f_channels=save_txt)
    log.info("Re-Built model using {}...".format(save_txt))
    if args.gpu:
        net.cuda()
    if args.load:
        net.load_state_dict(torch.load(save_file))
        log.info('Re-Loaded checkpoint from {}...'.format(save_file))

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    # Use epochs or iterations for fine-tuning
    save_file = osp.join(save_dir, "Finetuned.pth")

    finetune(net, optimizer, criterion, train, log, save_file,
             args.iters, args.epochs, args.batch_size, args.gpu, args.scale)

    val_dice = eval_net(net, valid,  len(valid), args.gpu, args.batch_size)
    log.info('Validation Dice Coeff: {}'.format(val_dice))


