import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch import optim
import logging
from data.dataset import get_dataset_and_dataloader
from model.layers import YAMLModel
from utils.loss import Loss
from test import Test
import wandb
logger = logging.getLogger(__name__)


class Train:
    def __init__(self, device, param, merge_category_flag = False, wandb_log=True, checkpoint = False):
        """
            :param device: cpu or gpu
            :param param:
            :param merge_category_flag: merge category or not
            :param wandb_log:
            :param checkpoint: load checkpoint or not
        """
        self.device = device
        self.batch_size = param['batch_size']
        self.param = param
        self.dataloader, self.dataset = get_dataset_and_dataloader(type='train', batch_size=param['batch_size'],
                                                                   img_resolution=param['img_resolution'],
                                                                   target_stride=param['target_stride'],
                                                                   gauss_kernel_sigma=param['gauss_kernel_sigma'],
                                                                   build_dataset=False, num_workers=param['num_workers'],
                                                                   merge_category_flag=merge_category_flag)
        self.class_num = self.dataset.class_num
        self.model = YAMLModel(param['model']).to(device)
        self.loss = Loss(device=device, movement_loss_weight=param['movement_loss_weight'], class_num=self.class_num)
        self.optimizer = optim.Adam(self.model.parameters(), lr=param['learning_rate'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=param['lr_scheduler_step_size'],
                                                   gamma=param['lr_scheduler_gamma'])
        self.test = Test(device, param, model=self.model, saved_model_name=None, merge_category_flag=merge_category_flag)
        self.wandb_log = wandb_log
        self.checkpoint = checkpoint
        if self.wandb_log:
            wandb.init(project="STTNET", config=param)
            wandb.watch(self.model)

    def run(self, num_epoch, model_save_name=None):
        if self.checkpoint:
            self.model.load(model_save_name)
        for epoch in range(num_epoch):
            self.model.train()
            logger.info("---------------------------------------------")
            logger.info(f"Epoch {epoch+1}")
            logger.info("---------------------------------------------")
            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            iter_per_epoch = int(len(self.dataset) // self.batch_size)
            pbar = tqdm(enumerate(self.dataloader), total=iter_per_epoch, desc="Training")
            for iter, (imgs, target_dist, target_movement, _) in pbar:
                self.optimizer.zero_grad()
                imgs = torch.tensor(imgs, device=device, dtype=torch.float32)
                pred = self.model(imgs)
                total_loss, center_loss, movement_loss = self.loss(pred, (target_dist, target_movement))
                total_loss.backward()
                self.optimizer.step()
                pbar.set_postfix({'Total loss': total_loss.item(), 'Center_loss': center_loss.item(),
                                  'Movement_loss': movement_loss.item()})
                if self.wandb_log:
                    wandb.log({"train_total_loss": total_loss, "train_center_loss": center_loss,
                               "train_movement_loss": movement_loss})
            self.scheduler.step()
            avg_total_loss, avg_center_loss, avg_movement_loss = self.test.run()
            logger.info(f"Total loss: {avg_total_loss:.6f}, Center loss: {avg_center_loss:.6f}, Movement loss: {avg_movement_loss:.6f}")
            if self.wandb_log:
                wandb.log({"test_total_loss": avg_total_loss, "test_center_loss": avg_center_loss,
                           "test_movement_loss": avg_movement_loss})
        if model_save_name:
            self.model.save(model_save_name)
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    parameter = {
        'batch_size': 2,
        'img_resolution': (640, 640),
        'target_stride': 4,
        'gauss_kernel_sigma': 3,
        'num_workers': 4,
        'model': 'hourglass',
        'movement_loss_weight': 1,
        'learning_rate': 0.0001,
        'lr_scheduler_step_size': 1,
        'lr_scheduler_gamma': 0.9
    }
    train = Train(device, parameter, merge_category_flag=True, wandb_log=True, checkpoint=False)
    train.run(num_epoch=1, model_save_name='test_1')

