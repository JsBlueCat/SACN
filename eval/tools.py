import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
# show dataset
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from collections import Counter


def count_accuracy_boxes(pred_boxes, gt_boxes, ep):
    tp, fp, tn, fn = 0, 0, 0, 0
    size = gt_boxes.shape[0]
    face_gt = np.zeros(size)
    reg_list = []
    for i in pred_boxes:
        temp = np.square((i-gt_boxes)[:, :2])
        temp = np.sqrt(temp[:, 0] + temp[:, 1])
        min_val = np.min(temp)
        if min_val > ep:
            fp += 1
            continue
        min_ind = np.squeeze(np.where(temp == np.min(min_val)))
        face_gt[min_ind] = 1
        tp += 1
        pred_rig = i[3]
        gt_rig = gt_boxes[min_ind, 3]
        det = get_angle_diff(pred_rig, gt_rig)
        reg_list.append(det)
    fn = size - np.sum(face_gt)
    # print(tp, fp, tn, fn)
    return tp, fp, tn, fn, reg_list


def get_angle_diff(a, b):
    d1 = a-b
    d2 = 2*180-d1
    if d1 > 0:
        d2 *= -1
    if d1 < d2:
        return d1
    else:
        return d2


def draw_roc_curve(fp, tpr):
    plt.figure()
    plt.plot(fp, tpr, '-', label='FDDB-up')
    plt.title('FDDB-up')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def draw_angle_curve(angle_list,rr):
    angle = list(int(e) for e in angle_list)
    result = dict().fromkeys(list(range(-180, 181)), 0)
    for a in angle:
        if a > 180:
            a -= 360
        elif a < -180:
            a += 360
        result[a] += 1
    names = np.array(list(result.keys()))
    values = np.array(list(result.values()))
    # print(names)
    # print(type(names))
    # ind = np.arange(len(names))  # the x locations for the groups
    width = 18  # the width of the bars
    names1 = np.array(list(rr.keys()))
    values1 = np.array(list(rr.values()))
    fig, ax = plt.subplots()

    rects2 = ax.bar(names1, values1, width,
                    color='IndianRed', label='UseNMS', alpha=1)
    rects1 = ax.bar(names, values, width,
                    color='SkyBlue', label='UseCC', alpha=0.1)
    # rects2 = ax.bar(ind + width/2, women_means, width,
    #                 color='IndianRed', label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency of Error 100 times')
    ax.set_title('Angle Error')
    ax.legend()
    # plt.xlim((-181,181))
    plt.show()
    print(result)


def draw_train_curve(file, size):
    with open(file, 'r') as f:
        datas = f.readlines()
    count = 0
    epoch_arr = []
    train_arr = []
    val_arr = []
    for data in datas:
        if count % 3 == 0:
            epoch = data.strip().split(' ')
            epoch_arr.append(epoch[0])
        elif count % 3 == 1:
            train = data.strip().split(' ')
            train_arr.append(float(train[1]))
        elif count % 3 == 2:
            val = data.strip().split(' ')
            val_arr.append(float(val[1]))
        count += 1

    x1 = list(range(0, size))
    x2 = list(range(0, size))
    y1 = train_arr
    y2 = val_arr

    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(x1, y1, x2, y2,'.-')
    # axs[0].set_xlabel('epoches')
    # axs[0].set_ylabel('Test and Train accuracy')
    # axs[0].grid(True)
    # fig.tight_layout()
    # plt.show()
    plt.figure()
    plt.plot(x1, y1, '-', label='Train Accuracy')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.plot(x2, y2, '-', label='Val Accuracy')
    # plt.xlabel('Test loss vs. epoches')
    # plt.ylabel('Test loss')
    plt.legend()
    plt.show()


def draw_train_curve3(file, size):
    with open(file, 'r') as f:
        datas = f.readlines()
    count = 0
    epoch_arr = []
    train_arr = []
    val_arr = []
    for data in datas:
        if count % 3 == 0:
            epoch = data.strip().split(' ')
            epoch_arr.append(epoch[0])
        elif count % 3 == 1:
            train = data.strip().split(' ')
            train_arr.append(float(train[1]))
        elif count % 3 == 2:
            val = data.strip().split(' ')
            val_arr.append(float(val[1]))
        count += 1

    x1 = list(range(0, size))
    x2 = list(range(0, size))
    y1 = train_arr
    y2 = val_arr

    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(x1, y1, x2, y2,'.-')
    # axs[0].set_xlabel('epoches')
    # axs[0].set_ylabel('Test and Train accuracy')
    # axs[0].grid(True)
    # fig.tight_layout()
    # plt.show()
    plt.figure()
    plt.plot(x1, y1, '-', label='Train mean angular error')
    plt.title('Mean angular error vs. epoches')
    plt.ylabel('Train accuracy')
    plt.plot(x2, y2, '-', label='Validate mean angular error')
    # plt.xlabel('Test loss vs. epoches')
    # plt.ylabel('Test loss')
    plt.legend()
    plt.show()

draw_train_curve('./SACN_2_val.txt', 500)
draw_train_curve3('./SACN_3_val.txt', 280)
