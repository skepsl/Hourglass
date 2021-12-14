from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import time

import utils.config as cnf

logger = logging.getLogger(__name__)
pd.set_option('display.max_rows', None)

class VisDroneMOT_DataSet(Dataset):
    col_names = ['frame_index', 'target_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
                 'score', 'object_category', 'truncation', 'occlusion']
    category_names = {0: 'ignored_region', 1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
                      6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11: 'others'}
    category_color = {0: (0, 0, 0), 1: (0, 255, 0), 2: (0, 128, 128), 3: (255, 0, 0), 4: (0, 0, 255),
                      5: (0, 255, 255), 6: (128, 0, 128), 7: (255, 255, 0), 8: (42, 42, 165), 9: (203, 192, 255),
                      10: (0, 165, 255), 11: (255, 255, 255)}

    def __init__(self, dataset_type, img_resolution=(640, 640), target_stride=4, gauss_kernel_sigma=1, merge_category_flag=False):
        self.dataset_type = dataset_type
        self.img_resolution = np.array(img_resolution)
        self.tgt_stride = target_stride
        self.tgt_resolution = (self.img_resolution / self.tgt_stride).astype(int)
        # file_dir = Path(__file__).parents[2].resolve()
        file_dir = Path('D:/').resolve()
        self.dataset_dir = file_dir / ('VisDrone_MOT/VisDrone2019-MOT-' + dataset_type)
        self.sequence_dirs = list((self.dataset_dir / 'sequences').glob('*'))
        self.annotation_files = list((self.dataset_dir / 'annotations').glob('*.txt'))
        self.num_videos = len(self.sequence_dirs)
        self.image = []
        self.label = []
        self.data_cache = Path(__file__).parent / ('dataset_' + self.dataset_type + '.cache')
        for video_index, (annotation_file, sequence_dir) in enumerate(zip(self.annotation_files, self.sequence_dirs)):
            img_file = pd.DataFrame(sequence_dir.glob('*.jpg'), columns=["image_file"])
            img_file.insert(0, "video_index", video_index)
            img_file.insert(1, "frame_index", np.arange(img_file.shape[0]))
            self.image.append(img_file)
            label = pd.read_csv(annotation_file, names=self.col_names)
            label["frame_index"] -= 1
            label.insert(0, "video_index", video_index)
            self.label.append(label)
        self.image = pd.concat(self.image, ignore_index=True)
        self.label = pd.concat(self.label, ignore_index=True)
        self.num_frame = self.label.pivot_table(index='video_index', columns=[], values='frame_index', aggfunc=np.max).to_numpy()[:, 0]
        self.num_target = self.label.pivot_table(index='video_index', columns=[], values='target_id', aggfunc=np.max).to_numpy()[:, 0]
        self.resolution = []
        for _, (_, _, img_file) in self.image.query('frame_index == 0').iterrows():
            dim = cv2.imread(str(img_file)).shape
            self.resolution.append([dim[1], dim[0]])
        self.resolution = np.array(self.resolution)
        self.video_statistics = None
        if self.data_cache.is_file():
            self.data = torch.load(self.data_cache)
        else:
            self.data = []
        # generate gaussian kernel
        xl = np.linspace(-self.tgt_resolution[0], self.tgt_resolution[0], self.tgt_resolution[0]*2 + 1).astype(float)
        yl = np.linspace(-self.tgt_resolution[1], self.tgt_resolution[1], self.tgt_resolution[1]*2 + 1).astype(float)
        xg, yg = np.meshgrid(xl, yl)
        s = float(gauss_kernel_sigma)
        self.gauss_kernel = np.exp(-(xg**2.0 + yg**2.0) / (2.0 * s**2.0))
        self.merge_category_flag = merge_category_flag
        if self.merge_category_flag:
            self.class_num = len(cnf.merge_category_dict)
        else:
            self.class_num = len(self.col_names)

    def show_video_statistics(self):
        num_box = self.label.pivot_table(index='video_index', columns='object_category', values='score',
                                         aggfunc=np.sum, fill_value=0, dropna=False)
        total_num_box = num_box.sum(axis=1)
        num_box = pd.concat((num_box, total_num_box), axis=1)
        resolution = pd.DataFrame([(lambda x: '(' + str(x[0]) + ',' + str(x[1]) + ')')(x) for x in self.resolution])
        video_info = pd.concat((resolution, pd.DataFrame(self.num_frame), pd.DataFrame(self.num_target)), axis=1)
        self.video_statistics = pd.concat((pd.DataFrame(video_info), num_box), axis=1)
        idx_list = [['resolution', 'num_frame', 'num_target'] + ['num_box'] * (len(self.category_names)+1),
                    ['', '', ''] + list(self.category_names.values()) + ['total']]
        idx = pd.MultiIndex.from_arrays(idx_list)
        self.video_statistics.columns = idx
        self.video_statistics.drop(('num_box', 'ignored_region'), axis=1, inplace=True)
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.width', 1000)
        logger.info(self.video_statistics)

    def show_video(self, video_idx_list=None):
        if video_idx_list is not None:
            video_idx_list = video_idx_list if isinstance(video_idx_list, list) else [video_idx_list]
            images = self.image.query('video_index in @video_idx_list')
        else:
            images = self.image
        logging.info("Start showing video")
        for _, (video_idx, frame_idx, img_file) in tqdm(images.iterrows(), total=images.shape[0]):
            img = cv2.imread(str(img_file))
            num_obj = 0
            for _, obj in self.label.query('(video_index == @video_idx) and (frame_index == @frame_idx)').iterrows():
                target_id = np.array(obj['target_id'])
                bbox = np.array(obj[['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']])
                object_category = self.category_names[obj['object_category']]
                object_color = self.category_color[obj['object_category']]
                top_left = bbox[0:2]
                bottom_right = top_left + bbox[2:]
                cv2.rectangle(img, top_left, bottom_right, object_color, 1)
                text = object_category + '(' + str(target_id) + ')'
                cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 1, cv2.LINE_AA)
                num_obj += 1
            text = 'Video: ' + str(video_idx) + ', Frame: ' + str(frame_idx) + ', Object: ' + str(num_obj)
            cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("Labeled video", img)
            cv2.waitKey(25)
        cv2.destroyAllWindows()

    def build_dataset(self, frame_num=1, frame_spacing=1, frame_stride=1, video_idx_list=None, obj_cat_list=None):
        self.data = []
        if video_idx_list is None:
            video_idx_list = list(range(self.num_videos))
        logging.info(f'Start building dataset for list of videos {video_idx_list}')
        for video_idx in video_idx_list:
            logging.info(f'Processing video {video_idx}: ')
            video_image = self.image.query('video_index == @video_idx')
            video_label = self.label.query('video_index == @video_idx')
            num_frame = self.num_frame[video_idx]
            frame_span = frame_num * frame_spacing
            frame_starts = range(0, num_frame-frame_span, frame_stride)
            for frame_start in tqdm(frame_starts):
                frame_idx_list = list(range(frame_start, frame_start+frame_span+1, frame_spacing))
                image_file_list = list(video_image.query('frame_index in @frame_idx_list')['image_file'])
                # build ignored region information
                ignored_region_list = []
                for frame_idx in frame_idx_list:
                    ignored_region = video_label.query('(frame_index == @frame_idx) and (object_category == 0)')
                    ignored_region = ignored_region[['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']].to_numpy()
                    ignored_region_list.append(ignored_region)
                start_frame_idx = frame_idx_list[0]
                start_labels = video_label.query('frame_index == @start_frame_idx')
                target_list = []
                for _, start_target in start_labels.iterrows():
                    start_target_id = start_target['target_id']
                    end_frame_idx = frame_idx_list[-1]
                    end_target = video_label.query('(frame_index == @end_frame_idx) and (target_id == @start_target_id)')
                    if len(end_target.index) > 0:
                        end_target = end_target.iloc[0]
                        obj_cat = start_target['object_category']
                        if ((obj_cat_list is not None) and (obj_cat not in obj_cat_list)) or obj_cat == 0:
                            continue
                        start_width = start_target['bbox_width']
                        start_height = start_target['bbox_height']
                        end_width = end_target['bbox_width']
                        end_height = end_target['bbox_height']
                        start_center_x = start_target['bbox_left'] + start_width / 2
                        start_center_y = start_target['bbox_top'] + start_height / 2
                        end_center_x = end_target['bbox_left'] + end_width / 2
                        end_center_y = end_target['bbox_top'] + end_height / 2
                        displacement_x = end_center_x - start_center_x
                        displacement_y = end_center_y - start_center_y
                        target_list.append(np.array([start_center_x, start_center_y, end_center_x, end_center_y,
                                                     displacement_x, displacement_y, start_width, start_height,
                                                     end_width, end_height, obj_cat]))
                if target_list:
                    targets = np.stack(target_list, axis=0)
                    self.data.append((image_file_list, ignored_region_list, targets))
        torch.save(self.data, self.data_cache)

    def show_dataset_image(self):
        logging.info("Start showing dataset")
        for idx in tqdm(range(len(self))):
            imgs, target_dist, target_movement = self[idx]
            first_img = (np.transpose(imgs[:3], (1, 2, 0)) * 255).astype(np.uint8)
            last_img = (np.transpose(imgs[-3:], (1, 2, 0)) * 255).astype(np.uint8)
            z = np.zeros_like(target_dist[0])
            first_target_dist = (np.stack((target_dist[0], z, z), axis=2) * 255).astype(np.uint8)
            last_target_dist = (np.stack((z, z, target_dist[1]), axis=2) * 255).astype(np.uint8)
            first_target_dist = cv2.resize(first_target_dist, dsize=self.img_resolution, interpolation=cv2.INTER_LINEAR)
            last_target_dist = cv2.resize(last_target_dist, dsize=self.img_resolution, interpolation=cv2.INTER_LINEAR)
            first_img = cv2.addWeighted(first_img, 0.5, first_target_dist, 0.5, 0.0)
            last_img = cv2.addWeighted(last_img, 0.5, last_target_dist, 0.5, 0.0)
            img = cv2.addWeighted(first_img, 0.5, last_img, 0.5, 0.0)
            for target in target_movement:
                first_center = (target[:2] * self.tgt_stride + self.tgt_stride/2.0).astype(int)
                displacement = (target[2:4] * self.tgt_stride).astype(int)
                cv2.arrowedLine(img, tuple(first_center), tuple(first_center+displacement), (0, 255, 0), 1)
            cv2.imshow("Dataset image", img)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

    def load_and_resize_image(self, img_files, ignored_region_list, targets):
        img_list = []
        for img_file, ignored_region in zip(img_files, ignored_region_list):
            tmp_img = cv2.imread(str(img_file))
            for box in ignored_region:
                tmp_img = cv2.rectangle(tmp_img, box[:2], box[:2] + box[2:4], (0,0,0), -1)
            img_list.append(tmp_img)
        if self.img_resolution is not None:
            cur_resolution = (img_list[0].shape[1], img_list[0].shape[0])
            for idx, img in enumerate(img_list):
                img_list[idx] = cv2.resize(img, dsize=self.img_resolution, interpolation=cv2.INTER_LINEAR)
            w_ratio = self.img_resolution[0]/cur_resolution[0]
            h_ratio = self.img_resolution[1]/cur_resolution[1]
            targets *= np.array([w_ratio, h_ratio, w_ratio, h_ratio, w_ratio, h_ratio, w_ratio, h_ratio,
                                 w_ratio, h_ratio, 1])
        for idx, img in enumerate(img_list):
            img_list[idx] = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        imgs = np.concatenate(img_list, axis=0)
        return imgs, targets

    def convert_target(self, targets):
        tgt_class = (targets[:, -1]).astype(int)
        if self.merge_category_flag:
            tgt_class = self.merge_category(tgt_class)

        start_center = (targets[:, :2] / self.tgt_stride).astype(int)
        end_center = (targets[:, 2:4] / self.tgt_stride).astype(int)
        displacement = (targets[:, 4:6] / self.tgt_stride).astype(float)
        tgt_resolution_x = self.tgt_resolution[0]
        tgt_resolution_y = self.tgt_resolution[1]
        target_dist = np.zeros((self.class_num*2, tgt_resolution_y, tgt_resolution_x))
        target_movement = np.zeros((len(tgt_class), self.class_num*4))

        for i in range(self.class_num):
            cls_idx = np.where(tgt_class == i+1)
            start_dist = np.zeros((tgt_resolution_y, tgt_resolution_x))
            end_dist = np.zeros((tgt_resolution_y, tgt_resolution_x))
            for sc, ec in zip(start_center[cls_idx], end_center[cls_idx]):
                x, y = int(tgt_resolution_x - sc[0]), int(tgt_resolution_y - sc[1])
                start_dist = np.maximum(start_dist, self.gauss_kernel[y:y+tgt_resolution_y, x:x+tgt_resolution_x])
                x, y = int(tgt_resolution_x - ec[0]), int(tgt_resolution_y - ec[1])
                end_dist = np.maximum(end_dist, self.gauss_kernel[y:y+tgt_resolution_y, x:x+tgt_resolution_x])
            target_dist[i*2:i*2+2] = np.stack((start_dist, end_dist))
            target_movement[cls_idx, i*4:i*4+4] = np.concatenate((start_center[cls_idx], displacement[cls_idx]), axis=1)

        return target_dist, target_movement, tgt_class

    @staticmethod
    def merge_category(tgt_class):
        category_dict = cnf.merge_category_dict
        merge_category_num = len(category_dict)
        for idx in range(1, merge_category_num+1):
            tgt_class = np.where(np.isin(tgt_class, np.array(category_dict[idx], dtype=np.int8))==True, idx, tgt_class)
        return tgt_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_files, ignored_region_list, targets = self.data[idx]
        imgs, targets = self.load_and_resize_image(img_files, ignored_region_list, targets)
        target_dist, target_movement, target_class = self.convert_target(targets)
        return imgs, target_dist, target_movement, target_class


def collate_fn(batch):
    # batch_img:
    #   (batch, frame, ch, y_size, x_size) numpy array
    #   The size of ch is 3 (color)
    # batch_target_dist (target distribution of center points):
    #   (batch, ch, y_size, x_size) numpy array
    #   The size of ch is 2 (first image, last image)
    # batch_target_movement:
    #   list (with batch size) of (object_number, movement_info) numpy array
    #   The size of movement_info is 4 (first_center_x, first_center_y, x_displacement, y_displacement)
    batch_img, batch_target_dist, batch_target_movement, batch_target_class = zip(*batch)
    return np.stack(batch_img), np.stack(batch_target_dist), list(batch_target_movement), list(batch_target_class)


def get_dataset_and_dataloader(type, batch_size, img_resolution=(640, 640), target_stride=4,
                               gauss_kernel_sigma=1, build_dataset=False, build_params=None, num_workers=0,
                               merge_category_flag=False):
    dataset = VisDroneMOT_DataSet(dataset_type=type, img_resolution=img_resolution, target_stride=target_stride,
                                  gauss_kernel_sigma=gauss_kernel_sigma, merge_category_flag=merge_category_flag)
    if build_dataset:
        dataset.build_dataset(frame_num=build_params.frame_num, frame_spacing=build_params.frame_spacing,
                              frame_stride=build_params.frame_stride, video_idx_list=build_params.video_idx_list,
                              obj_cat_list=build_params.obj_cat_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            collate_fn=collate_fn)
    return dataloader, dataset


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    # create dataset
    # dataset = VisDroneMOT_DataSet(dataset_type='train', img_resolution=(640, 640), target_stride=4, gauss_kernel_sigma=3)
    # dataset = VisDroneMOT_DataSet(dataset_type='test-dev', img_resolution=(640, 640), target_stride=4, gauss_kernel_sigma=3)

    # show statistics and labeled video
    #dataset.show_video_statistics()
    # dataset.show_video(video_idx_list=None)
    # dataset.show_video(video_idx_list=[15])
    # build and show dataset (for stacked images)
    # object categories
    # 0: 'ignored_region', 1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11: 'others'
    #obj_cat_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #dataset.build_dataset(frame_num=7, frame_spacing=1, frame_stride=1, video_idx_list=None, obj_cat_list=obj_cat_list)
    # dataset.show_dataset_image()

    # setup dataloader and test performance
    num_workers = 4
    dataloader, dataset = get_dataset_and_dataloader(type='train', batch_size=16, img_resolution=(640, 640),
                                                    target_stride=4, gauss_kernel_sigma=3, build_dataset=False,
                                                    num_workers=num_workers, merge_category_flag=True)
    # dataset.show_video_statistics()
    # dataset.show_video()
    # start_time = time.time()
    batch_count = 0
    for idx, (img, dist, movement, cls) in tqdm(enumerate(dataloader)):
        batch_count += 1
    # total_elapsed_time = time.time() - start_time
    # batch_per_second = batch_count / total_elapsed_time
    # logger.info(f"Batch per second: {batch_per_second:.3f}Hz")
