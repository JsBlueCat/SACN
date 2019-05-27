from math import log
import torch
import torch.nn as nn


class Wing_loss(nn.Module):
    def __init__(self, w=10, epsilon=2):
        super(Wing_loss, self).__init__()
        self._w = w
        self._epsilon = epsilon
        # C = w − w ln(1 + w/e)
        self._C = w - w * log(1 + w / epsilon)

    def forward(self, input):
        '''
        wing(x) =   w ln(1 + |x|/e) if |x| < w
                    |x| − C         otherwise
        '''
        x_abs = torch.abs(input)
        wing = torch.where(x_abs < self.w, self.w *
                           torch.log(1+x_abs.div_(self.epsilon)), x_abs - self._C)
        loss = wing.float().mean()
        return loss


class LossFn_1(nn.Module):
    def __init__(self, face_factor=1, box_factor=0.5, rig_factor=0.5):
        super(LossFn_1, self).__init__()
        self.face_factor = face_factor
        self.box_factor = box_factor
        self.rig_factor = rig_factor
        self.face_loss_fn = nn.BCELoss()
        # for SACN they use smooth l1 loss
        self.box_loss_fn = nn.MSELoss()
        # for wing they use wing loss
        # self.box_loss_fn = Wing_loss(w=10, epsilon=2)
        self.rig_loss_fn = nn.BCELoss()

    def face_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.face_loss_fn(valid_pred_label, valid_gt_label)*self.face_factor

    # def face_loss(self, gt_label, pred_label):
    #     pred_label = torch.squeeze(pred_label)
    #     gt_label = torch.squeeze(gt_label)

    #     # the pred_label is implement by sigmoid function
    #     # so we get the element which great equal than 0
    #     # we pick the positive and negetive face to train
    #     # face detector
    #     mask = torch.ge(gt_label, 0)
    #     valid_gt_label = torch.masked_select(gt_label, mask)
    #     valid_pred_label = torch.masked_select(pred_label, mask)
    #     return self.face_loss_fn(valid_pred_label, valid_gt_label)*self.face_factor

    # def box_loss(self, gt_label, box_target, box_pred):
    #     '''
    #     the target contains 3 factors
    #     ta tb tw  represents the regression
    #     for smoothl1loss we define ta tb tw to minimize its
    #     train loss
    #     box_target is a nD-tensor like [[1,2,10],...]
    #     '''
    #     box_target = torch.squeeze(box_target)
    #     box_pred = torch.squeeze(box_pred)
    #     gt_label = torch.squeeze(gt_label)
    #     # remove the negtive face
    #     # first we choose the negtive face and set
    #     # its index with 1 other 0 and we choose other
    #     unmask = torch.eq(gt_label, 0)
    #     mask = torch.eq(unmask, 0)

    #     choose_index = torch.nonzero(mask.data)
    #     choose_index = torch.squeeze(choose_index)

    #     valid_box_target = box_target[choose_index, :]
    #     valid_box_pred = box_pred[choose_index, :]
    #     # only face can effect the loss
    #     return self.box_loss_fn(valid_box_pred, valid_box_target)*self.box_factor
    def box_loss(self, gt_label, gt_offset, pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)

        # get the mask element which != 0
        unmask = torch.eq(gt_label, 0)
        mask = torch.eq(unmask, 0)
        # convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        # only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index, :]
        valid_pred_offset = pred_offset[chose_index, :]
        return self.box_loss_fn(valid_pred_offset, valid_gt_offset)*self.box_factor

    def rig_loss(self, gt_label, target_rig, pred_rig):
        '''
        to train the rig_loss we choose the face and suspect face 
        '''
        pred_rig = torch.squeeze(pred_rig)
        target_rig = torch.squeeze(target_rig)
        gt_label = torch.squeeze(gt_label)
        '''
        only face has rig element and can effect the loss
        '''
        unmask = torch.eq(gt_label, 0)
        mask = torch.eq(unmask, 0)

        choose_index = torch.nonzero(mask.data)
        choose_index = torch.squeeze(choose_index)
        valid_pred_rig = pred_rig[choose_index]
        valid_gt_rig = target_rig[choose_index]

        # filter out not train rig
        mask = torch.ge(valid_gt_rig, 0)
        valid_gt_rig = torch.masked_select(valid_gt_rig, mask)
        valid_pred_rig = torch.masked_select(valid_pred_rig, mask)

        return self.rig_loss_fn(valid_pred_rig, valid_gt_rig)*self.rig_factor

    def forward(self, gt_label, pred_label, box_target, box_pred, target_rig, pred_rig):
        return self.face_loss(gt_label, pred_label) + self.box_loss(gt_label, box_target, box_pred) + self.rig_loss(gt_label, target_rig, pred_rig)


class LossFn_2(nn.Module):
    def __init__(self, face_factor=1, box_factor=0.5, rig_factor=0.5):
        super(LossFn_2, self).__init__()
        self.face_factor = face_factor
        self.box_factor = box_factor
        self.rig_factor = rig_factor
        self.face_loss_fn = nn.BCELoss()
        # for SACN they use smooth l1 loss
        self.box_loss_fn = nn.SmoothL1Loss()
        # for wing they use wing loss
        # self.box_loss_fn = Wing_loss(w=10, epsilon=2)
        self.rig_loss_fn = nn.CrossEntropyLoss()

    def face_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # the pred_label is implement by sigmoid function
        # so we get the element which great equal than 0
        # we pick the positive and negetive face to train
        # face detector
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.face_loss_fn(valid_pred_label, valid_gt_label)*self.face_factor

    def box_loss(self, gt_label, box_target, box_pred):
        '''
        the target contains 3 factors
        ta tb tw  represents the regression  
        for smoothl1loss we define ta tb tw to minimize its 
        train loss
        box_target is a nD-tensor like [[1,2,10],...]
        '''
        box_target = torch.squeeze(box_target)
        box_pred = torch.squeeze(box_pred)
        gt_label = torch.squeeze(gt_label)
        # remove the negtive face
        # first we choose the negtive face and set
        # its index with 1 other 0 and we choose other
        unmask = torch.eq(gt_label, 0)
        mask = torch.eq(unmask, 0)

        choose_index = torch.nonzero(mask.data)
        choose_index = torch.squeeze(choose_index)

        valid_box_target = box_target[choose_index, :]
        valid_box_pred = box_pred[choose_index, :]
        # only face can effect the loss
        return self.box_loss_fn(valid_box_pred, valid_box_target)*self.box_factor

    def rig_loss(self, gt_label, target_rig, pred_rig):
        '''
        to train the rig_loss we choose the face and suspect face 
        '''
        pred_rig = torch.squeeze(pred_rig)
        target_rig = torch.squeeze(target_rig)
        gt_label = torch.squeeze(gt_label)
        # _, pred_rig = torch.max(pred_rig, dim=1)
        '''
        only face has rig element and can effect the loss
        '''
        index = torch.nonzero(gt_label)
        index = torch.squeeze(index)
        valid_gt_rig = torch.index_select(target_rig, dim=0, index=index)
        valid_pred_rig = torch.index_select(pred_rig, dim=0, index=index)
        mask = valid_gt_rig >= 0
        index = torch.nonzero(mask)
        index = torch.squeeze(index)
        valid_gt_rig = torch.index_select(valid_gt_rig, dim=0, index=index)
        valid_gt_rig = valid_gt_rig.long()
        valid_pred_rig = torch.index_select(valid_pred_rig, dim=0, index=index)
        return self.rig_loss_fn(valid_pred_rig, valid_gt_rig)*self.rig_factor

    def forward(self, gt_label, pred_label, box_target, box_pred, target_rig, pred_rig):
        return self.face_loss(gt_label, pred_label) + self.box_loss(gt_label, box_target, box_pred) + self.rig_loss(gt_label, target_rig, pred_rig)


class LossFn_3(nn.Module):
    def __init__(self, face_factor=1, box_factor=0.5, rig_factor=0.5):
        super(LossFn_3, self).__init__()
        self.face_factor = face_factor
        self.box_factor = box_factor
        self.rig_factor = rig_factor
        self.face_loss_fn = nn.BCELoss()
        # for SACN they use smooth l1 loss
        self.box_loss_fn = nn.SmoothL1Loss()
        # for wing they use wing loss
        # self.box_loss_fn = Wing_loss(w=10, epsilon=2)
        self.rig_loss_fn = nn.SmoothL1Loss()

    def face_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # the pred_label is implement by sigmoid function
        # so we get the element which great equal than 0
        # we pick the positive and negetive face to train
        # face detector
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.face_loss_fn(valid_pred_label, valid_gt_label)*self.face_factor

    def box_loss(self, gt_label, box_target, box_pred):
        '''
        the target contains 3 factors
        ta tb tw  represents the regression  
        for smoothl1loss we define ta tb tw to minimize its 
        train loss
        box_target is a nD-tensor like [[1,2,10],...]
        '''
        box_target = torch.squeeze(box_target)
        box_pred = torch.squeeze(box_pred)
        gt_label = torch.squeeze(gt_label)
        # remove the negtive face
        # first we choose the negtive face and set
        # its index with 1 other 0 and we choose other
        unmask = torch.eq(gt_label, 0)
        mask = torch.eq(unmask, 0)

        choose_index = torch.nonzero(mask.data)
        choose_index = torch.squeeze(choose_index)

        valid_box_target = box_target[choose_index, :]
        valid_box_pred = box_pred[choose_index, :]
        # only face can effect the loss
        return self.box_loss_fn(valid_box_pred, valid_box_target)*self.box_factor

    def rig_loss(self, gt_label, target_rig, pred_rig):
        '''
        to train the rig_loss we choose the face and suspect face 
        '''
        pred_rig = torch.squeeze(pred_rig)
        target_rig = torch.squeeze(target_rig)
        gt_label = torch.squeeze(gt_label)
        # _, pred_rig = torch.max(pred_rig, dim=1)
        '''
        only face has rig element and can effect the loss
        '''
        index = torch.nonzero(gt_label)
        index = torch.squeeze(index)
        valid_gt_rig = torch.index_select(target_rig, dim=0, index=index)
        valid_pred_rig = torch.index_select(pred_rig, dim=0, index=index)
        return self.rig_loss_fn(valid_pred_rig, valid_gt_rig)*self.rig_factor

    def forward(self, gt_label, pred_label, box_target, box_pred, target_rig, pred_rig):
        return self.face_loss(gt_label, pred_label) + self.box_loss(gt_label, box_target, box_pred) + self.rig_loss(gt_label, target_rig, pred_rig)
