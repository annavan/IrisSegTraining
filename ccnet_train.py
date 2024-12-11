import numpy as np
import torch
import os
import csv
import math
import random
import glob
import sys

from PIL import Image
from argparse import ArgumentParser
from skimage import transform
from scipy.stats.mstats import mquantiles
import torch.nn as nn

from torch.optim import SGD, Adam, AdamW, NAdam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize
from torch.nn import functional as F

from modules.dataset import IrisSegmDataset, test_time_augment, merge_results
from modules.network import UNet, NestedUNet, NestedResUNet, NestedDenseUNet, GCN, DeepLab, SegNet, SegResNet, NestedAtrousResUNet, NestedSharedAtrousResUNet, NestedSharedAtrousAttentionResUNetIN
from modules.criterion import CrossEntropyLoss2d, PixelWiseIoULoss, PixelWiseDIoULoss, DiceLoss, InvDiceLoss, CompleteDiceLoss
from modules.transform import Relabel, ToLabel
import madgrad
from pytorch_optimizer import Prodigy

NUM_CHANNELS = 1
NUM_CLASSES = 1

output_feature = []

image_transform = ToPILImage()

image_mean_std = (0.5791223733793273, 0.21176097694558188)
mask_mean_std = (0.17338686627523558, 0.35294113772891444)

input_transform = Compose([
    ToTensor(),
    Normalize(mean=(0.5,), std=(0.5,))
])

target_transform = Compose([
    ToTensor()
])

def iou_score(output, target, need_sigmoid=True):
    smooth = 1e-10       
    
    if torch.is_tensor(output):
        if need_sigmoid:
            output = torch.sigmoid(output)
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    
    #print(output_.shape, target_.shape)
    
    intersection = np.sum((output_ & target_), axis=(1,2))
    union = np.sum((output_ | target_), axis=(1,2))
    
    iou_indiv = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou_indiv)

    return iou

def train(args, model):
    
    if args.log_txt:
        sys.stdout = open('./' + args.model_type.lower() + '_' + str(args.width) + '_' + args.tag + '_trainer_output.txt', 'a')
        
    print(model)
    
    directory = os.path.dirname(args.model_type.lower() + '_' + str(args.width) + '_' + args.tag + '_checkpoint/')
    if not os.path.exists(directory):
        os.makedirs(directory)
   
    #weight = torch.ones(NUM_CLASSES)
    print(f"image_dir: {args.image_dir}")
    print(f"mask_dir: {args.mask_dir}")
    print(f"input_transform: {input_transform}")
    print(f"target_transform: {target_transform}")
    print("------------loader beginning-----------") 
    dataset = IrisSegmDataset(args.image_dir, args.mask_dir, True, input_transform, target_transform)
    val_length = int(0.05 * len(dataset))
    train_length = len(dataset) - val_length
    val_dataset, train_dataset = random_split(dataset, [val_length, train_length], generator=torch.Generator().manual_seed(42))
    print('Train Dataset Length:', len(train_dataset), ', Val Dataset Length:', len(val_dataset))
    loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    print("loader complete")
    print("-----------val loader beginning--------")
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    print("val_loader complete")
    
    
    print('training length:', len(loader))
    print('validation length:', len(val_loader))
    print("softmax calculated")

    if args.loss_type == 'cross_entropy':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == 'cross_entropy+dice':
        criterion = nn.BCEWithLogitsLoss()
        criterion2 = CompleteDiceLoss()
    elif args.loss_type == 'dice':
        criterion = CompleteDiceLoss()
    else:
        print('Print please select a valid loss type.')
        exit()
    
    print('Using', args.loss_type)
    
    if args.optim_type == 'sgd':
        optimizer = SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.001)
    elif args.optim_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optim_type == 'madgrad':
        optimizer = madgrad.MADGRAD(model.parameters(), lr=args.lr, weight_decay=0.001)
    elif args.optim_type == 'prodigy':
        optimizer = Prodigy(model.parameters())
    
    best_val_loss_average = float('inf')
    
    if args.cuda:
        scaler = torch.GradScaler("cuda")

    for epoch in range(1, args.num_epochs+1):

        model.train()
        
        #if (epoch - 1) % 10 == 0 and epoch != 1:
        #    for param_group in optimizer.param_groups:
        #        param_group["lr"] *= 0.1

        epoch_loss = []
        train_IoU = 0
        #train_IoU_tta = 0
        
        loader.dataset.dataset.set_augment(True)
        for batch, data in enumerate(loader):
            # setup input
            images = data["image"]
            labels = data["mask"]
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            # push through network and optimize
            inputs = Variable(images)
            targets = Variable(labels)
            
            if args.label_smoothing:
                rand_1 = torch.rand(targets.shape)
                rand_0 = torch.rand(targets.shape)
                soft_targets_1 = torch.ones(targets.shape)-rand_1*0.2
                soft_targets_0 = rand_0*0.2
                if args.cuda:
                    soft_targets_0 = soft_targets_0.cuda()
                    soft_targets_1 = soft_targets_1.cuda()
                targets = torch.where(targets>0.5, soft_targets_1, soft_targets_0).requires_grad_(False)
                
            
            optimizer.zero_grad(set_to_none=True)
            if args.cuda:
                with torch.autocast("cuda"):
                    outputs = model(inputs)
                    if args.loss_type == 'cross_entropy+dice':
                        loss = 0.5 * criterion(outputs, targets.reshape(outputs.shape)) + 0.5 * criterion2(outputs, targets.reshape(outputs.shape))
                    else:
                        loss = criterion(outputs, targets.reshape(outputs.shape))
            else:
                outputs = model(inputs)
                if args.loss_type == 'cross_entropy+dice':
                    loss = 0.5 * criterion(outputs, targets.reshape(outputs.shape)) + 0.5 * criterion2(outputs, targets.reshape(outputs.shape))
                else:
                    loss = criterion(outputs, targets.reshape(outputs.shape))
                    
            if args.cuda:                            
                scaler.scale(loss).backward()
                if args.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
            
            # compute IoU
            IoU = iou_score(outputs.reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]), targets.reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]))
            train_IoU += IoU
            
            #IoU_tta = iou_score(outputs_tta.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]), targets.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]))
            #train_IoU_tta += IoU_tta
 
            epoch_loss.append(loss.item())
            
            if batch % args.log_batch== 0:
                train_loss_average = sum(epoch_loss) / len(epoch_loss)
                print("Train loss: {aver} (epoch: {epoch}, batch: {batch}, IoU: {IoU})".format(aver = train_loss_average, epoch = epoch, batch = batch, IoU = train_IoU/args.log_batch))
                #print("Train loss: {aver} (epoch: {epoch}, batch: {batch}, IoU: {IoU}, IoU TTA: {IoU_tta})".format(aver = train_loss_average, epoch = epoch, batch = batch, IoU = train_IoU/args.log_batch, IoU_tta = train_IoU_tta/args.log_batch))
                train_IoU = 0
                #train_IoU_tta = 0
                if args.log_txt:
                    sys.stdout.close()
                    sys.stdout = open('./' + args.model_type.lower() + '_' + str(args.width) + '_' + args.tag + '_trainer_output.txt', 'a')

        # Evaluate test images
        #if epoch % args.eval_epoch == 0:
        #    evaluate(args, model, epoch)

        # Validation set
        if len(val_loader) > 0:
            val_loader.dataset.dataset.set_augment(False)
            val_epoch_loss = []
            val_IoU = 0
            #val_IoU_tta = 0
            model.eval()
            with torch.no_grad():
                for batch, data in enumerate(val_loader):
                    # setup input
                    images = data["image"]
                    labels = data["mask"]
                    if args.cuda:
                        images = images.cuda()
                        labels = labels.cuda()
                    
                    # push through network and compute loss
                    inputs = Variable(images)
                    targets = Variable(labels)

                    outputs = model(inputs)
                    if args.loss_type == 'cross_entropy+dice':
                        val_loss = 0.5 * criterion(outputs, targets.reshape(outputs.shape)) + 0.5 * criterion2(outputs, targets.reshape(outputs.shape))
                        #val_loss = 0.4 * criterion(outputs, targets.reshape(outputs.shape)) + 0.4 * SoftDiceLoss()(outputs, targets.reshape(outputs.shape)) + 0.2 * SoftInvDiceLoss()(outputs, targets.reshape(outputs.shape))
                    else:
                        val_loss = criterion(outputs, targets.reshape(outputs.shape))
                    val_epoch_loss.append(val_loss.item())       
                    
                    # compute IoU
                    IoU = iou_score(outputs.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]), targets.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]))
                    val_IoU += IoU
                
                val_loss_average = sum(val_epoch_loss) / len(val_epoch_loss)
                val_IoU /= len(val_loader)
                print("Val loss: {aver} (epoch: {epoch}), Val IoU: {val_IoU}".format(aver = val_loss_average, epoch = epoch, val_IoU = val_IoU))
                #print("Val loss: {aver} (epoch: {epoch}), Val IoU: {val_IoU}".format(aver = val_loss_average, epoch = epoch, val_IoU = val_IoU))
                if args.log_txt:
                    sys.stdout.close() 
                    sys.stdout = open('./' + args.model_type.lower() + '_' + str(args.width) + '_' + args.tag + '_trainer_output.txt', 'a')
                if val_loss_average < best_val_loss_average:
                    # Save checkpoint
                    best_val_loss_average = val_loss_average
                    filename = os.path.join(directory, "{model}-{epoch:03}-{val}-maskIoU-{val_IoU}.pth".format(model = args.model_type, epoch = epoch, val=round(val_loss_average, 6), val_IoU = round(val_IoU, 6)))
                    #filename = os.path.join(directory, "{model}-{epoch:03}-{val}-maskIoU-{val_IoU}.pth".format(model = args.model_type, epoch = epoch, val=round(val_loss_average, 6), val_IoU = round(val_IoU, 6)))
                    if args.multi_gpu:
                        torch.save(model.module.state_dict(), filename)
                    else:
                        torch.save(model.state_dict(), filename)
    if args.log_txt:
        sys.stdout.close()
        
def evaluate(args, model):
    val_loader.dataset.dataset.set_augment(False)
    val_epoch_loss = []
    val_IoU = 0
    val_IoU_tta = 0
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(val_loader):
            # setup input
            images = data["image"]
            labels = data["mask"]
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            
            # push through network and compute loss
            inputs = Variable(images)
            targets = Variable(labels)
            
            outputs_tta = []
            for i in range(inputs.shape[0]):
                imgs, masks, aug_list = test_time_augment(inputs[i])
                aug_outputs = []
                for img in imgs:
                    output = torch.sigmoid(model(img.unsqueeze(0)))[0]
                    aug_outputs.append(output)
                pred = merge_results(aug_outputs, masks, aug_list)
                outputs_tta.append(pred)
            outputs_tta = torch.stack(pred, dim=0)  
            
            outputs = model(inputs)
            if args.loss_type == 'cross_entropy+dice':
                val_loss = 0.4 * criterion(outputs, targets.reshape(outputs.shape)) + 0.4 * SoftDiceLoss()(outputs, targets.reshape(outputs.shape)) + 0.2 * SoftInvDiceLoss()(outputs, targets.reshape(outputs.shape))
            else:
                val_loss = criterion(outputs, targets.reshape(outputs.shape))
            val_epoch_loss.append(val_loss.item())       
            
            # compute IoU
            IoU = iou_score(outputs.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]), targets.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]))
            val_IoU += IoU
            
            IoU_tta = iou_score(outputs_tta.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]), targets.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]))
            val_IoU_tta += IoU_tta
            
        val_IoU /= len(val_loader)
        val_IoU_tta /= len(val_loader)
        print("Val loss: {aver} (epoch: {epoch}), Val IoU: {val_IoU}, Val IoU tta: {val_IoU_tta}".format(aver = val_loss_average, epoch = epoch, val_IoU = val_IoU, val_IoU_tta = val_IoU_tta))
        
'''        
def evaluate(args, model):
    print("evaluating...")
    image_root = "./data/input/"
    save_root = "./data/output/"
    with open('dataset_ac.csv', 'rb') as f:
        lines = f.readlines()
    meta = list(map(lambda l: [i.rstrip('\n') for i in l.decode("utf-8").split(',')], lines))
    meta = meta[1:]

    model.eval()

    softmax = nn.LogSoftmax(dim=1)
    for m in meta:
        print(m[0][2:])
        # Fetch input image
        image_path = image_root + m[0][2:]
        image = Image.open(image_path).convert('L')
        image = image.resize((320, 240), Image.BILINEAR)
        image = input_transform(image).unsqueeze(0)
        if args.cuda:
            image = image.cuda()
        image = Variable(image)

        # Run through the network
        print(image.shape)
        outputs = model(image)
        logprob = softmax(outputs).data.cpu().numpy()
        pred = np.argmax(logprob, axis=1)*255
        im = Image.fromarray(pred[0].astype(np.uint8))
        out_path = save_root + m[0][2:].split('/')[-1].split('.')[0] + '_mask.bmp'
        im.save(out_path)

    print("Test set images saved!")

    return None
'''
def main(args):
    if args.model_type.lower() == 'unet':
        model = UNet(NUM_CLASSES, NUM_CHANNELS)
    elif args.model_type.lower() == 'nestedunet':
        model = NestedUNet(NUM_CLASSES, NUM_CHANNELS, width=args.width)
    elif args.model_type.lower() == 'nestedresunet':
        model = NestedResUNet(NUM_CLASSES, NUM_CHANNELS, width=args.width)
    elif args.model_type.lower() == 'nesteddenseunet':
        model = NestedDenseUNet(NUM_CLASSES, NUM_CHANNELS, width=args.width)
    elif args.model_type.lower() == 'gcn':
        model = GCN(NUM_CLASSES)
    elif args.model_type.lower() == 'segnet':
        model = SegNet(NUM_CLASSES, in_channels=1)
    elif args.model_type.lower() == 'segresnet':
        model = SegResNet(NUM_CLASSES, in_channels=1)
    elif args.model_type.lower() == 'deeplab':
        model = DeepLab(NUM_CLASSES, in_channels=1)
    elif args.model_type.lower() == 'nestedatrousresunet':
        model = NestedAtrousResUNet(NUM_CLASSES, NUM_CHANNELS, width=args.width)
    elif args.model_type.lower() == 'nestedsharedatrousresunet':
        model = NestedSharedAtrousResUNet(NUM_CLASSES, NUM_CHANNELS, width=args.width)
    elif args.model_type.lower() == 'nestedsharedatrousresunetsimple':
        model = NestedSharedAtrousResUNetSimple(NUM_CLASSES, NUM_CHANNELS, width=args.width)
    elif args.model_type.lower() == 'nestedsharedatrousattentionresunetin':
        model = NestedSharedAtrousAttentionResUNetIN(NUM_CLASSES, NUM_CHANNELS, width=args.width)
        
        
    if args.state:
        try:
            if args.cuda:
                model.load_state_dict(torch.load(args.state, map_location=torch.device('cuda')))
            else:
                model.load_state_dict(torch.load(args.state, map_location=torch.device('cpu')))
            print("model state loaded")
        except AssertionError:
            print("assertion error")
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
    
    if args.cuda:
        if torch.cuda.device_count() > 1 and args.multi_gpu:
            model = nn.DataParallel(model.cuda())
        else:
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
            model = model.cuda()
        if args.cudnn:
            print('Using CUDNN')
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True        

    if args.mode == 'eval':
        evaluate(args, model)
    if args.mode == 'train':
        train(args, model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--state')
    parser.add_argument('--mode', default = "train")
    parser.add_argument('--image_dir')
    parser.add_argument('--mask_dir')
    parser.add_argument('--model_type', type=str, default='nestedsharedatrousattentionresunetin')
    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--num_epochs', type=int, default=2001)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_batch', type=int, default=10)
    parser.add_argument('--loss_type', type=str, default='cross_entropy+dice')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--log_txt', action='store_true')
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--tag', type=str, default='finemasks')
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--optim_type', type=str, default='madgrad')
    parser.add_argument('--clip_grad', action='store_true')
    parser.add_argument('--max_norm', type=float, default=0.1)

    main(parser.parse_args())

