import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import logging
import data.dataset as dataset
import model.layers as layers
import utils.config as cnf

logger = logging.getLogger(__name__)


class Loss:
    def __init__(self, device, movement_loss_weight=1.0, max_object=200, class_num=len(cnf.merge_category_dict)):
        self.center_loss_func = self.center_loss
        self.movement_loss_func = self.movement_loss
        self.device = device
        self.movement_loss_weight = movement_loss_weight
        self.max_object = max_object
        self.class_num = class_num

    def center_loss(self, pred, target):
        pos_idx = target.eq(1).float()
        neg_idx = target.lt(1).float()
        neg_weight = torch.pow(1 - target, 4)
        pred = torch.clamp(nn.Sigmoid()(pred), min=1e-4, max=(1 - 1e-4))
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_idx
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weight * neg_idx
        num_pos = pos_idx.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            loss = - neg_loss
        else:
            loss = - (pos_loss + neg_loss) / num_pos
        return loss

    def movement_loss(self, pred, target, mask):
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(target).float()
        pred = pred * mask
        target = target * mask
        loss = F.smooth_l1_loss(pred, target, reduction='sum')
        loss = loss / (num + 1e-4)
        return loss

    def get_movement_pred_and_target_tensor(self, pred, target):
        batch_size = len(target)
        new_pred = torch.zeros((batch_size, self.max_object, self.class_num*2)).to(self.device) # (1, 200, 6)
        new_target = torch.zeros((batch_size, self.max_object, self.class_num*2)).to(self.device)
        mask = torch.zeros((batch_size, self.max_object)).to(self.device)
        for i in range(self.class_num):
            for idx, tgt in enumerate(target): # (#, 12)
                num_obj = min(tgt.shape[0], self.max_object)
                tgt_x_idx = tgt[:num_obj, i*4] # (#, )
                tgt_y_idx = tgt[:num_obj, i*4+1] # (#, )
                tgt_displacement = tgt[:num_obj, i*4+2:i*4+4] # (#, 2)
                new_pred[idx, :num_obj, i*2:i*2+2] = torch.transpose(pred[idx, i*2:i*2+2, tgt_y_idx, tgt_x_idx], 0, 1)
                new_target[idx, :num_obj, i*2:i*2+2] = torch.Tensor(tgt_displacement).to(self.device)
                mask[idx, :num_obj] = torch.ones(num_obj).to(self.device)
        return new_pred, new_target, mask

    def __call__(self, pred, target):
        # pred:
        #   (batch, ch, y_size, x_size) tensor
        #   The size of ch is 4 (first image, last image, x movement, y movement)
        # target (target_dist, target_movement):
        #   target_dist (target distribution of center points):
        #       (batch, ch, y_size, x_size) numpy array
        #       The size of ch is 2 (first image, last image)
        #   target_movement:
        #       list (with batch size) of (object_number, movement_info) numpy array
        #       The size of movement_info is 4 (first_center_x, first_center_y, x_displacement, y_displacement)
        target_dist, target_movement = target
        pred_center = pred[:, :self.class_num*2, :, :]
        pred_movement = pred[:, self.class_num*2:, :, :]
        target_center = torch.tensor(target_dist, device=self.device, dtype=torch.float32) # (b, 6, 160, 160)
        center_loss = self.center_loss_func(pred_center, target_center)
        pred_movement, target_movement, mask = self.get_movement_pred_and_target_tensor(pred_movement, target_movement)
        movement_loss = self.movement_loss_func(pred_movement, target_movement, mask)
        total_loss = center_loss + self.movement_loss_weight * movement_loss
        return total_loss, center_loss, movement_loss


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')


    num_workers = 0
    dataloader, dataset = dataset.get_dataset_and_dataloader(type='train', batch_size=1, img_resolution=(640, 640),
                                                             target_stride=4, gauss_kernel_sigma=4,
                                                             build_dataset=False, num_workers=num_workers, merge_category_flag=True)
    class_num = dataset.class_num
    print(class_num)
    loss = Loss(device=device, movement_loss_weight=1, class_num=class_num)
    model = layers.YAMLModel('hourglass').to(device)
    imgs, target_dist, target_movement, _ = next(iter(dataloader))
    imgs = torch.tensor(imgs, device=device, dtype=torch.float32)
    pred = model(imgs)
    total_loss, center_loss, movement_loss = loss(pred, (target_dist, target_movement))
    total_loss.backward()
