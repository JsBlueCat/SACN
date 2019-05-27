import os
import sys
import time
import glob
import numpy as np
import torch
sys.path.append("/home/cery/workspace/darts-new/SACN")
from loss import *
import utils
import logging
import logging.handlers
import datetime
import argparse
import torch.nn as nn
import torch.utils
import torchvision
import wider_dset as dset
import torch.backends.cudnn as cudnn
from collections import namedtuple


from torch.autograd import Variable
from model import Network_SACN_1_fcn as Network

# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
# SACN_1 = genotype = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1)], normal_concat=range(
#     2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 3), ('dil_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

parser = argparse.ArgumentParser(description="SACN_1 train model")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.05, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=500,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=5,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='SACN_1',
                    help='which architecture to use')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--stage', type=int, default=1, help='train SACN stage')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logger = logging.getLogger('mylogger')
# logger.setLevel(logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

rf_handler = logging.handlers.TimedRotatingFileHandler(
    'all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
rf_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"))

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
fh.setLevel(logging.INFO)

fh1 = logging.FileHandler(os.path.join(args.save, 'val.txt'))
fh1.setFormatter(logging.Formatter("%(message)s"))
fh1.setLevel(logging.WARN)

logger.addHandler(rf_handler)
logger.addHandler(fh)
logger.addHandler(fh1)


# logger = logging.getLogger('mylogger')
# logger.setLevel(logging.DEBUG)

# f_handler = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# f_handler.setLevel(logging.INFO)
# f_handler.setFormatter(logging.Formatter(log_format))

# kq_handler = logging.FileHandler(os.path.join(args.save, 'val.txt'))
# kq_handler.setLevel(logging.WARN)
# kq_handler.setFormatter(logging.Formatter("%(message)s"))

# logger.addHandler(f_handler)
# logger.addHandler(kq_handler)


def main():
    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logger.info('gpu device = %d' % args.gpu)
    logger.info("args = %s", args)

    # genotype = eval("%s" % args.arch)
    # model = Network(args.init_channels, args.layerls, genotype)
    model = Network()
    model = model.cuda()
    # print('111')
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))
    # print('222')
    face_factor, bbox_reg_factor, rig_factor = 1, 0.8, 0.8
    criterion = LossFn_1(face_factor=face_factor,
                         box_factor=bbox_reg_factor, rig_factor=rig_factor)
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_Wider(args)
    train_data = dset.wider(
        anno_file="../anno_store/imglist_anno_24_180.txt", train=True, transform=train_transform)
    valid_data = dset.wider(
        anno_file="../anno_store/imglist_anno_24_180.txt", train=False, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=12)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs))

    for epoch in range(args.epochs):
        scheduler.step()
        logger.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        # logger.info('epoch %d', epoch)
        # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        logger.warn('%d', epoch)

        train_acc, train_rig, train_bbx, train_obj = train(
            train_queue, model, criterion, optimizer)
        logger.info('train_label_acc %f rig %f bbx %f',
                    train_acc, train_rig, train_bbx)

        logger.warn('%f %f %f', train_acc, train_rig, train_bbx)

        valid_acc, valid_rig, valid_bbx, valid_obj = infer(
            valid_queue, model, criterion)
        logger.info('valid_label_acc %f rig %f bbx %f',
                    valid_acc, valid_rig, valid_bbx)

        logger.warn('%f %f %f', valid_acc, valid_rig, valid_bbx)

        utils.save(model, os.path.join(args.save, 'weights%d.pt' % epoch))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgRageCtrl()
    face = utils.AvgRageCtrl()
    rig_avg = utils.AvgRageCtrl()
    reg = utils.AvgRageCtrl()
    model.train()

    for step, (input, label, rig, bbox) in enumerate(train_queue):
        input = Variable(input, requires_grad=False).cuda()
        label = Variable(label.float(), requires_grad=False).cuda(async=True)
        bbox = Variable(bbox.float(), requires_grad=False).cuda(async=True)
        rig = Variable(rig.float(), requires_grad=False).cuda(async=True)

        optimizer.zero_grad()
        label_pred, bbx_pred, rig_pred = model(input)
        loss = criterion(label, label_pred, bbox, bbx_pred, rig, rig_pred)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = utils.compute_sigmoid_accuracy(label_pred, label)
        prec2 = utils.compute_rig_accuracy(label, rig_pred, rig)
        prec3 = criterion.box_loss(label, bbox, bbx_pred)
        n = input.size(0)
        objs.update(loss.data[0], n)
        face.update(prec1.data[0], n)
        rig_avg.update(prec2.data[0], n)
        reg.update(prec3.data[0], n)

        if step % args.report_freq == 0:
            logger.info('train %03d %e %f %f %f', step,
                        objs.avg, face.avg, rig_avg.avg, reg.avg)

    return face.avg, rig_avg.avg, reg.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgRageCtrl()
    face = utils.AvgRageCtrl()
    rig_avg = utils.AvgRageCtrl()
    reg = utils.AvgRageCtrl()
    model.eval()

    for step, (input, label, rig, bbox) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        label = Variable(label.float(), volatile=True).cuda(async=True)
        bbox = Variable(bbox.float(), volatile=True).cuda(async=True)
        rig = Variable(rig.float(), volatile=True).cuda(async=True)

        label_pred, bbx_pred, rig_pred = model(input)
        loss = criterion(label, label_pred, bbox, bbx_pred, rig, rig_pred)

        prec1 = utils.compute_sigmoid_accuracy(label_pred, label)
        prec2 = utils.compute_rig_accuracy(label, rig_pred, rig)
        prec3 = criterion.box_loss(label, bbox, bbx_pred)
        n = input.size(0)
        objs.update(loss.data[0], n)
        face.update(prec1.data[0], n)
        rig_avg.update(prec2.data[0], n)
        reg.update(prec3.data[0], n)
        if step % args.report_freq == 0:
            logger.info('valid  %03d %e %f %f %f', step,
                        objs.avg, face.avg, rig_avg.avg, reg.avg)

    return face.avg, rig_avg.avg, reg.avg, objs.avg


if __name__ == '__main__':
    main()
