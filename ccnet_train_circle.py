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

from torch.optim import SGD, Adam, AdamW, NAdam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize
from torch.nn import functional as F
import torchvision
from torch.cuda.amp import GradScaler

from modules.dataset import IrisSegmDatasetCircle
from modules.network import *
from modules.transform import Relabel, ToLabel
import madgrad

from pytorch_optimizer import Prodigy

from torchvision import models
from torchvision.ops import generalized_box_iou_loss

from tqdm import tqdm

import cv2

NUM_CHANNELS = 1
NUM_PARAMS = 6


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

def iou_score_params(inp_xyrs, tar_xyrs):

    Y, X = np.ogrid[:240, :320]

    inp_masks_list = []
    for i in range(inp_xyrs.shape[0]):
        xyr = inp_xyrs[i]
        dist_from_center_pupil = np.sqrt((X - (xyr[0]*320))**2 + (Y - (xyr[1]*240))**2)
        pupil_mask = dist_from_center_pupil <= (xyr[2]*160)
        dist_from_center_iris = np.sqrt((X - (xyr[3]*320))**2 + (Y - (xyr[4]*240))**2)
        iris_mask = dist_from_center_iris <= (xyr[5]*200)
        mask = np.where(pupil_mask == True, False, iris_mask)
        inp_masks_list.append(mask)
    
    inp_masks = np.stack(inp_masks_list)
    
    tar_masks_list = []
    for i in range(tar_xyrs.shape[0]):
        xyr = tar_xyrs[i]
        dist_from_center_pupil = np.sqrt((X - (xyr[0]*320))**2 + (Y - (xyr[1]*240))**2)
        pupil_mask = dist_from_center_pupil <= (xyr[2]*160)
        dist_from_center_iris = np.sqrt((X - (xyr[3]*320))**2 + (Y - (xyr[4]*240))**2)
        iris_mask = dist_from_center_iris <= (xyr[5]*200)
        mask = np.where(pupil_mask == True, False, iris_mask)
        tar_masks_list.append(mask)

    tar_masks = np.stack(tar_masks_list)
    
    smooth = 1e-10

    intersection = np.sum((inp_masks & tar_masks), axis=(1,2))
    union = np.sum((inp_masks | tar_masks), axis=(1,2))
    
    iou_indiv = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou_indiv)
    
    return iou

def train(args, model):
    
    if args.log_txt:
        sys.stdout = open('./' + args.model_type.lower() + '_' + str(args.width) + '_' + args.tag + '_trainer_output.txt', 'a')
        
    print(model)

    # Confirm CUDA setup
    print("CUDA Available:", torch.cuda.is_available())
    print(torch.__version__)
    print("gradscaler: ", hasattr(torch, 'GradScaler'))
    #print torch cuda is available if torch.cuda.is_available() os true
    print("torch: ", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Using GPU Device:", torch.cuda.current_device())
        print("CUDA Device Count:", torch.cuda.device_count())
    
    directory = os.path.dirname(args.model_type.lower() + '_' + str(args.width) + '_' + args.tag + '_checkpoint/')
    if not os.path.exists(directory):
        os.makedirs(directory)
   
    #weight = torch.ones(NUM_CLASSES)
    print(f"input_transform: {input_transform}")
    print("------------loader beginning-----------") 
    dataset = IrisSegmDatasetCircle(args.param_csv, True, input_transform, args.train_on_image)
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
    
    criterion = nn.SmoothL1Loss(reduction='sum')

    

    if torch.cuda.device_count() > 1 and args.multi_gpu:
        if args.optim_type == 'prodigy':
            optimizer = Prodigy(model.module.parameters(), weight_decay=args.weight_decay)
        elif args.optim_type == 'adamw':
            optimizer = AdamW(model.module.parameters(), args.lr, weight_decay=args.weight_decay)
        elif args.optim_type == 'madgrad':
            optimizer = madgrad.MADGRAD(model.module.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        if args.optim_type == 'prodigy':
            optimizer = Prodigy(model.parameters(), weight_decay=args.weight_decay)
        elif args.optim_type == 'adamw':
            optimizer = AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
        elif args.optim_type == 'madgrad':
            optimizer = madgrad.MADGRAD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    best_val_IoU = 0.0
    
    if args.cuda:
        scaler = GradScaler()
    
    for epoch in range(1, args.num_epochs+1):

        model.train()

        epoch_loss = []
        train_IoU = 0
        #train_IoU_tta = 0
        
        loader.dataset.dataset.set_augment(True)
        for batch, data in enumerate(loader):
            # setup input
            images = data["image"]
            if args.repeat_channel:
                images = images.repeat(1,3,1,1)
            circle_params = data["circle_param"]
            if args.cuda:
                images = images.cuda()
                circle_params = circle_params.cuda()

            # push through network and optimize
            inputs = Variable(images)
            targets = Variable(circle_params)
            
            optimizer.zero_grad(set_to_none=True)
            if args.cuda:
                with torch.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.reshape(outputs.shape))
            else:
                outputs = model(inputs)
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
            IoU = iou_score_params(outputs.clone().detach().cpu().numpy(), targets.reshape(outputs.shape).clone().detach().cpu().numpy())
            train_IoU += IoU
            
            epoch_loss.append(loss.item())
            
            if batch % args.log_batch== 0:
                train_loss_average = sum(epoch_loss) / len(epoch_loss)
                print("Train loss: {aver} (epoch: {epoch}, batch: {batch}, IoU: {IoU})".format(aver = train_loss_average, epoch = epoch, batch = batch, IoU = train_IoU/args.log_batch))
                
                train_IoU = 0
                
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
                    if args.repeat_channel:
                        images = images.repeat(1,3,1,1)
                    circle_params = data["circle_param"]
                    if args.cuda:
                        images = images.cuda()
                        circle_params = circle_params.cuda()
                    
                    # push through network and compute loss
                    inputs = Variable(images)
                    targets = Variable(circle_params)

                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets.reshape(outputs.shape))
                    val_epoch_loss.append(val_loss.item())       
                    
                    # compute IoU
                    IoU = iou_score_params(outputs.clone().detach().cpu().numpy(), targets.reshape(outputs.shape).clone().detach().cpu().numpy())
                    #IoU = iou_score(outputs.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]), targets.clone().detach().requires_grad_(False).reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]))
                    val_IoU += IoU
                
                val_loss_average = sum(val_epoch_loss) / len(val_epoch_loss)
                val_IoU /= len(val_loader)
                print("Val loss: {aver} (epoch: {epoch}), Val IoU: {val_IoU}".format(aver = val_loss_average, epoch = epoch, val_IoU = val_IoU))
                #print("Val loss: {aver} (epoch: {epoch}), Val IoU: {val_IoU}".format(aver = val_loss_average, epoch = epoch, val_IoU = val_IoU))
                if args.log_txt:
                    sys.stdout.close() 
                    sys.stdout = open('./' + args.model_type.lower() + '_' + str(args.width) + '_' + args.tag + '_trainer_output.txt', 'a')
                if val_IoU > best_val_IoU:
                    # Save checkpoint
                    best_val_IoU = val_IoU
                    filename = os.path.join(directory, "{model}-{epoch:03}-{val}-maskIoU-{val_IoU}.pth".format(model = args.model_type, epoch = epoch, val=round(val_loss_average, 6), val_IoU = round(val_IoU, 6)))
                    #filename = os.path.join(directory, "{model}-{epoch:03}-{val}-maskIoU-{val_IoU}.pth".format(model = args.model_type, epoch = epoch, val=round(val_loss_average, 6), val_IoU = round(val_IoU, 6)))
                    if args.multi_gpu:
                        torch.save(model.module.state_dict(), filename)
                    else:
                        torch.save(model.state_dict(), filename)
    if args.log_txt:
        sys.stdout.close()
        
def check_data(args):
    #weight = torch.ones(NUM_CLASSES)
    print(f"input_transform: {input_transform}")
    print("------------loader beginning-----------") 
    dataset = IrisSegmDatasetCircle(args.param_csv, True, input_transform)
    val_length = int(0.05 * len(dataset))
    train_length = len(dataset) - val_length
    val_dataset, train_dataset = random_split(dataset, [val_length, train_length], generator=torch.Generator().manual_seed(42))
    print('Train Dataset Length:', len(train_dataset), ', Val Dataset Length:', len(val_dataset))
    loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    print("loader complete")
    print("-----------val loader beginning--------")
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    print("val_loader complete")
    
    if not os.path.exists("./debug_dataset/"):
        os.mkdir("./debug_dataset/")
    
    if not os.path.exists("./get_data/"):
        os.mkdir("./get_data/")
    
    for batch, data in tqdm(enumerate(loader)):
        images = data["image"]
        circle_params = data["circle_param"]
        for i in range(images.shape[0]):
            imVis = np.stack((images[i][0].cpu().numpy() * 255,)*3, axis=-1)
            imVis = cv2.circle(imVis, (int(circle_params[i][0].item() * 320), int(circle_params[i][1].item() * 240)) , int(circle_params[i][2].item() * 160), (0, 0, 255), 2)
            imVis = cv2.circle(imVis, (int(circle_params[i][3].item() * 320), int(circle_params[i][4].item() * 240)) , int(circle_params[i][5].item() * 200), (255, 0, 0), 2)
            cv2.imwrite("./debug_dataset/" + str(batch) + "_" + str(i) + ".png", imVis)
            #Image.fromarray(np.uint8(images[i][0].cpu().numpy() * 255), 'L').save("./get_data/" + str(batch) + "_" + str(i) + "_" + str(circle_params[i].tolist()) +".png")
    
    
def main(args):
    args.repeat_channel = False
    if args.model_type.lower() == 'sharedatrousresnetparam':
        model = SharedAtrousResNetParam(NUM_PARAMS, NUM_CHANNELS, width=args.width, blocks_per_resolution=args.bpr)
    elif args.model_type.lower() == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.avgpool = conv(in_channels=512, out_n=6)
        model.fc = fclayer(out_n=6) 
        args.repeat_channel = True
    elif args.model_type.lower() == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.avgpool = conv(in_channels=512, out_n=6)
        model.fc = fclayer(out_n=6)
        args.repeat_channel = True
    elif args.model_type.lower() == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.avgpool = conv(in_channels=2048, out_n=6)
        model.fc = fclayer(out_n=6)
        args.repeat_channel = True
    elif args.model_type.lower() == 'convnext_tiny':
        model = models.convnext_tiny(weights='DEFAULT')
        model.avgpool = conv(in_channels=768, out_n=6)
        model.classifier = fclayer(in_h=7, in_w=10, out_n=6)
        args.repeat_channel = True
        args.width = 768
    elif args.model_type.lower() == 'convnext_small':
        model = models.convnext_small(weights='DEFAULT')
        model.avgpool = conv(in_channels=768, out_n=6)
        model.classifier = fclayer(in_h=7, in_w=10, out_n=6) 
        args.repeat_channel = True
        args.width = 768
    elif args.model_type.lower() == 'convnext_base':
        model = models.convnext_base(weights='DEFAULT')
        model.avgpool = conv(in_channels=1024, out_n=6)
        model.classifier = fclayer(in_h=7, in_w=10, out_n=6)
        args.repeat_channel = True
        args.width = 1024
    elif args.model_type.lower() == 'convnext_large':
        model = models.convnext_large(weights='DEFAULT')
        model.avgpool = conv(in_channels=1536, out_n=6)
        model.classifier = fclayer(in_h=7, in_w=10, out_n=6)
        args.repeat_channel = True
        args.width = 1536
    elif args.model_type.lower() == 'yolo_v8_n':
        model = yolo_v8_n(num_classes = 6)
    elif args.model_type.lower() == 'yolo_v8_s':
        model = yolo_v8_s(num_classes = 6)
    elif args.model_type.lower() == 'yolo_v8_m':
        model = yolo_v8_m(num_classes = 6)
    elif args.model_type.lower() == 'yolo_v8_l':
        model = yolo_v8_l(num_classes = 6)
    elif args.model_type.lower() == 'yolo_v8_x':
        model = yolo_v8_x(num_classes = 6)
        
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

    if not args.check_data:
        train(args, model)
    else:
        check_data(args)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--state')
    parser.add_argument('--param_csv')
    parser.add_argument('--model_type', type=str, default='sharedatrousresnetparam')
    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--num_epochs', type=int, default=5001)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_batch', type=int, default=10)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--log_txt', action='store_true')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--tag', type=str, default='circleparams')
    parser.add_argument('--check_data', action='store_true')
    parser.add_argument('--train_on_image', action='store_true')
    parser.add_argument('--clip_grad', action='store_true')
    parser.add_argument('--max_norm', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--optim_type', type=str, default='prodigy')

    main(parser.parse_args())
