import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from data.dataset import get_dataset_and_dataloader
from model.layers import YAMLModel
from utils.loss import Loss
from utils.helper import find_peak, rematch_vector
from utils.metrics import Metric
import utils.config as cnf
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


class Test:
    def __init__(self, device, param, merge_category_flag = False, model=None, saved_model_name=None):
        self.device = device
        self.param = param
        self.dataloader, self.dataset = get_dataset_and_dataloader(type='test-dev', batch_size=param['batch_size'],
                                                                   img_resolution=param['img_resolution'],
                                                                   target_stride=param['target_stride'],
                                                                   gauss_kernel_sigma=param['gauss_kernel_sigma'],
                                                                   build_dataset=False, num_workers=param['num_workers'],
                                                                   merge_category_flag=merge_category_flag)
        self.class_num = self.dataset.class_num
        self.num_data = len(self.dataset)
        self.num_batch = len(self.dataloader)
        self.img_resolution = self.dataset.img_resolution
        self.tgt_stride = self.dataset.tgt_stride
        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = YAMLModel(param['model']).to(device)
        if saved_model_name:
            self.model.load(saved_model_name)
        self.loss = Loss(device=device, movement_loss_weight=param['movement_loss_weight'], class_num=self.class_num)
        self.metric = Metric(self.device, self.dataset.tgt_resolution, self.param['batch_size'], self.class_num)

    @torch.no_grad()
    def run(self):
        pbar = tqdm(enumerate(self.dataloader), total=self.num_batch, desc="Testing")
        self.model.eval()
        avg_total_loss, avg_center_loss, avg_movement_loss = 0, 0, 0
        count = 0
        for iter, (imgs, target_dist, target_movement, _) in pbar:
            imgs = torch.tensor(imgs, device=self.device, dtype=torch.float32)
            pred = self.model(imgs)
            total_loss, center_loss, movement_loss = self.loss(pred, (target_dist, target_movement))
            pbar.set_postfix({'Total loss': total_loss.item(), 'Center_loss': center_loss.item(),
                              'Movement_loss': movement_loss.item()})
            avg_total_loss += total_loss
            avg_center_loss += center_loss
            avg_movement_loss += movement_loss
            count += 1
        avg_total_loss /= count
        avg_center_loss /= count
        avg_movement_loss /= count
        return avg_total_loss, avg_center_loss, avg_movement_loss

    @torch.no_grad()
    def run_metric(self):
        pbar=tqdm(enumerate(self.dataloader), total = self.num_batch, desc = "Testing")
        self.model.eval()
        count=0
        first_avg_metric=np.zeros((3,), dtype = np.float32)  # oks, mAP, f1
        last_avg_metric=np.zeros((3,), dtype = np.float32)  # oks, mAP, f1
        cosine_avg_metric=0
        for iter, (imgs, target_dist, target_movement, _) in pbar:
            imgs=torch.tensor(imgs, device = self.device, dtype = torch.float32)
            pred=self.model(imgs)
            first_peak_metrics, last_peak_metrics, cosine_metrics=self.metric.get_batch_statistics(pred, (
            target_dist, target_movement))
            cosine=np.sum(cosine_metrics)/self.class_num
            cosine_avg_metric+=cosine
            first_oks, _, _, first_mAP, first_f1=np.sum(first_peak_metrics, axis = 0)/self.class_num
            first_avg_metric+=np.array([first_oks, first_mAP, first_f1])
            last_oks, _, _, last_mAP, last_f1=np.sum(last_peak_metrics, axis = 0)/self.class_num
            last_avg_metric+=np.array([last_oks, last_mAP, last_f1])
            pbar.set_postfix({'first_oks': first_oks, 'first_mAP': first_mAP, 'first_f1': first_f1,
                              'last_oks': last_oks, 'last_mAP': last_mAP, 'last_f1': last_f1, 'cosine': cosine})
            count+=1
        first_avg_metric/=count
        last_avg_metric/=count
        cosine_avg_metric/=count
        logger.info(
            f"first_avg_oks: {first_avg_metric[0]:.6f}, first_avg_mAP: {first_avg_metric[1]:.6f}, first_avg_f1: {first_avg_metric[2]:.6f}\n"
            f"last_avg_oks: {last_avg_metric[0]:.6f}, last_avg_mAP: {last_avg_metric[1]:.6f}, last_avg_f1: {last_avg_metric[2]:.6f}\n"
            f"cosine_similarity: {cosine_avg_metric:.6f}")

    @torch.no_grad()
    def get_result_image(self, dataset_idx=None, peak_threshold=None, img_save_file=None, show_image=False):
        if dataset_idx is None:
            dataset_idx = list(range(self.num_data))
        else:
            dataset_idx = dataset_idx if isinstance(dataset_idx, list) else [dataset_idx]
        self.model.eval()
        img_list = []
        for idx in tqdm(dataset_idx, desc="processing result image"):
            imgs, _, _ = self.dataset[idx]
            imgs_tensor = torch.tensor(imgs[np.newaxis, :, :, :], device=self.device, dtype=torch.float32)
            pred = self.model(imgs_tensor)
            center_pred = nn.Sigmoid()(pred[0, :2, :, :]).cpu().numpy()
            movement_pred = pred[0, 2:4, :, :].cpu().numpy()
            first_img = (np.transpose(imgs[:3], (1, 2, 0)) * 255).astype(np.uint8)
            last_img = (np.transpose(imgs[-3:], (1, 2, 0)) * 255).astype(np.uint8)
            z = np.zeros_like(center_pred[0])
            first_dist = (np.stack((center_pred[0], z, z), axis=2) * 255).astype(np.uint8)
            last_dist = (np.stack((z, z, center_pred[1]), axis=2) * 255).astype(np.uint8)
            first_dist = cv2.resize(first_dist, dsize=self.img_resolution, interpolation=cv2.INTER_LINEAR)
            last_dist = cv2.resize(last_dist, dsize=self.img_resolution, interpolation=cv2.INTER_LINEAR)
            first_img = cv2.addWeighted(first_img, 0.5, first_dist, 0.5, 0.0)
            last_img = cv2.addWeighted(last_img, 0.5, last_dist, 0.5, 0.0)
            img = cv2.addWeighted(first_img, 0.5, last_img, 0.5, 0.0)
            first_img_peak, last_img_peak = find_peak(center_pred[:2], threshold=peak_threshold)
            displacement_xy = (np.transpose(movement_pred[:, first_img_peak[:, 0], first_img_peak[:, 1]]) * self.tgt_stride).astype(int)
            first_img_peak_xy = np.fliplr(first_img_peak) * self.tgt_stride + np.array([int(self.tgt_stride/2), int(self.tgt_stride/2)])
            last_img_peak_xy = np.fliplr(last_img_peak) * self.tgt_stride + np.array([int(self.tgt_stride/2), int(self.tgt_stride/2)])
            for p, d in zip(first_img_peak_xy, displacement_xy):
                cv2.circle(img, tuple(p), 1, (255, 0, 0), -1)
                cv2.arrowedLine(img, tuple(p), tuple(p+d), (0, 255, 0), 1)
            for p in last_img_peak_xy:
                cv2.circle(img, tuple(p), 1, (0, 0, 255), -1)
            img_list.append(img)
        if img_save_file:
            for idx, img in enumerate(img_list):
                file_name = str(Path(__file__).parent / 'test_result' / (img_save_file + '_' + str(idx) + '.jpg'))
                cv2.imwrite(file_name, img)
        if show_image:
            for idx, img in tqdm(enumerate(img_list), desc="showing result image"):
                cv2.imshow("Result image", img)
                cv2.waitKey(500)
        return img_list

    @torch.no_grad()
    def get_rematch_vector_image(self, dataset_idx=None, peak_threshold=None, img_save_file=None, split=False):
        if dataset_idx is None:
            dataset_idx=list(range(self.num_data))
        else:
            dataset_idx=dataset_idx if isinstance(dataset_idx, list) else [dataset_idx]

        self.model.eval()
        img_list=[]
        before_img_list = []
        after_img_list = []
        for idx in tqdm(dataset_idx, desc = 'rematching vector'):
            imgs, target_dist, target_movement, _=self.dataset[idx]
            imgs_tensor=torch.tensor(imgs[np.newaxis, :, :, :], device = self.device, dtype = torch.float32)
            pred=self.model(imgs_tensor)

            first_img=(np.transpose(imgs[:3], (1, 2, 0))*255).astype(np.uint8)
            last_img=(np.transpose(imgs[-3:], (1, 2, 0))*255).astype(np.uint8)
            before_img=cv2.addWeighted(first_img, 0.5, last_img, 0.5, 0.0)
            after_img=cv2.addWeighted(first_img, 0.5, last_img, 0.5, 0.0)

            center_pred=nn.Sigmoid()(pred[0, :self.class_num*2, :, :]).cpu().numpy()
            movement_pred=pred[0, self.class_num*2:, :, :].cpu().numpy()

            for cls in range(0, self.class_num):
                cls_color=cnf.merge_category_color[cls + 1]
                cls_color_light=tuple(int(c + (w - c)*3/5) for c, w in zip(cls_color, (255, 255, 255)))
                cls_color_complementary=tuple(int(w - c) for c, w in zip(cls_color, (255, 255, 255)))

                _, _, before_pred_first_peak_xy, before_pred_last_peak_xy, before_pred_displacement_xy= \
                self.calculate_feak_each_class(center_pred[cls*2:cls*2 + 2, :, :], movement_pred[cls*2:cls*2 + 2, :, :],peak_threshold)

                after_pred_first_peak, after_pred_last_peak, _= \
                rematch_vector(center_pred[cls*2:cls*2 + 2, :, :], movement_pred[cls*2:cls*2 + 2, :, :], peak_threshold)

                after_first_peak_xy=np.fliplr(after_pred_first_peak)*self.tgt_stride + np.array(
                    [int(self.tgt_stride/2), int(self.tgt_stride/2)])
                after_last_peak_xy=np.fliplr(after_pred_last_peak)*self.tgt_stride + np.array(
                    [int(self.tgt_stride/2), int(self.tgt_stride/2)])

                for p, d in zip(before_pred_first_peak_xy, before_pred_displacement_xy):
                    cv2.circle(before_img, tuple(p), 1, cls_color_light, -1)
                    cv2.arrowedLine(before_img, tuple(p), tuple(p + d), cls_color_light, 1)
                for p in before_pred_last_peak_xy:
                    cv2.circle(before_img, tuple(p), 1, cls_color, -1)
                for f, l in zip(after_first_peak_xy, after_last_peak_xy):
                    if len(after_first_peak_xy) == 0 or len(after_last_peak_xy) == 0:
                        break
                    cv2.arrowedLine(after_img, tuple(f), tuple(l), cls_color_light, 1)
                    cv2.circle(after_img, tuple(f), 1, cls_color_light, -1)
                    cv2.circle(after_img, tuple(l), 1, cls_color, -1)
            before_img_list.append(before_img)
            after_img_list.append(after_img)
            img_list.append(cv2.hconcat([before_img, after_img]))
        if img_save_file and split is False:
            for idx, img in enumerate(img_list):
                file_name=str(Path(__file__).parent/'post_processing_result'/(img_save_file + '_' + str(idx) + '.jpg'))
                cv2.imwrite(file_name, img)
        else:
            for idx, (before_img, after_img) in enumerate(zip(before_img_list, after_img_list)):
                before_file_name=str(Path(__file__).parent/'test_result'/(img_save_file + '_' + str(idx) + '.jpg'))
                after_file_name = str(Path(__file__).parent/'post_processing_result'/(img_save_file + '_' + str(idx) + '.jpg'))
                cv2.imwrite(before_file_name, before_img)
                cv2.imwrite(after_file_name, after_img)

        return img_list

    def calculate_feak_each_class(self, center_pred, movement_pred, peak_threshold):
        z = np.zeros_like(center_pred[0])
        first_dist = (np.stack((center_pred[0], z, z), axis=2) * 255).astype(np.uint8)
        last_dist = (np.stack((z, z, center_pred[1]), axis=2) * 255).astype(np.uint8)
        first_dist = cv2.resize(first_dist, dsize=self.img_resolution, interpolation = cv2.INTER_LINEAR)
        last_dist = cv2.resize(last_dist, dsize = self.img_resolution, interpolation = cv2.INTER_LINEAR)
        first_img_peak, last_img_peak = find_peak(center_pred[:2], threshold=peak_threshold)
        displacement_xy = (np.transpose(movement_pred[:, first_img_peak[:,0], first_img_peak[:,1]]) * self.tgt_stride).astype(int)
        first_img_peak_xy = np.fliplr(first_img_peak) * self.tgt_stride + np.array([int(self.tgt_stride/2), int(self.tgt_stride/2)])
        last_img_peak_xy = np.fliplr(last_img_peak) * self.tgt_stride + np.array([int(self.tgt_stride/2), int(self.tgt_stride/2)])
        return first_dist, last_dist, first_img_peak_xy, last_img_peak_xy, displacement_xy

if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    parameter = {
        'batch_size': 8,
        'img_resolution': (640, 640),
        'target_stride': 4,
        'gauss_kernel_sigma': 3,
        'num_workers': 8,
        'model': 'hourglass',
        'movement_loss_weight': 1,
    }
    test = Test(device, parameter, model=None, saved_model_name='test_1')
    test.get_result_image(dataset_idx=None, peak_threshold=0.2, show_image=False, img_save_file='test_1')
