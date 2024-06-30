import random
from pathlib import Path
from time import gmtime, strftime

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm
from PIL import Image
from data import collect_data
import wandb
from matplotlib import pyplot
from collections import OrderedDict
import warnings
from typing import List, Tuple
from torchvision.models.detection.anchor_utils import AnchorGenerator

def preprocessing_for_resnet50():
    """
    Приведение изображения под формат ResNet50 (см https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
    :return:
    """
    preprocessing = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return preprocessing


class SiameseModel(nn.Module):
    """
    Сиамская сеть с backbone ResNet50 и активацией
    """
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50()
        backbone = nn.Sequential(*tuple(model.children())[:-2])
        out_features = 2048
        backbone.out_channels = out_features
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128,),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        self.model = torchvision.models.detection.FasterRCNN(backbone=backbone, num_classes=20,
                                                             rpn_anchor_generator=anchor_generator)

    def forward(self, big_images, reference_image, targets=None):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in big_images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        big_images, targets = self.model.transform(big_images, targets)
        reference_image, _ = self.model.transform(reference_image, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )
        features_big = self.model.backbone(big_images.tensors)
        features_ref = self.model.backbone(reference_image.tensors)
        features = torch.abs(features_big - features_ref)
        # do concat
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.model.rpn(big_images, features, targets)
        detections, detector_losses = self.model.roi_heads(features, proposals, big_images.image_sizes, targets)
        detections = self.model.transform.postprocess(detections, big_images.image_sizes,
                                                      original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections


class SiameseDataset(IterableDataset):
    """
    Датасет для сиамской сети
    Принимает коллекции изображений на вход
    Также все изображения проходят preprocessing
    Лучше бы сделать с SAHI
    """
    def __init__(self, data_coll, device='cpu'):
        self.images_paths_set, self.annot_set = data_coll
        self.device = device
        self.objects = {}
        for image_path, annots in zip(self.images_paths_set, self.annot_set):
            image = Image.open(image_path)
            for name,  (xmin, ymin), (xmax, ymax) in annots:
                bbox = (xmin, ymin, xmax, ymax)
                cropped = image.crop(bbox)
                if name not in self.objects.keys():
                    self.objects[name] = [cropped]
                else:
                    self.objects[name].append(cropped)
            image.close()

        self.preprocessing = preprocessing_for_resnet50()

    def compose_iterations(self):
        keys = list(self.objects.keys())
        while True:
            image_ind = np.random.randint(0, len(self.images_paths_set))
            image_path = self.images_paths_set[image_ind]
            image = Image.open(image_path)
            w, h = image.size
            annot = self.annot_set[image_ind]

            labels = np.array([i[0] for i in annot])
            chosen_label = np.random.choice(np.unique(labels))
            bboxes = torch.tensor([np.array([i[1][0] / w * 224, i[1][1] / h * 224, i[2][0] / w * 224, i[2][1] / h * 224])
                                  for i in annot], dtype=torch.float32, device=self.device)[labels == chosen_label]
            d = {}
            d['boxes'] = bboxes
            d['labels'] = torch.tensor([keys.index(i) for i in labels], dtype=torch.int64, device=self.device)
            target = [d]
            ref_image_ind = np.random.randint(0, len(self.objects[chosen_label]))
            ref_image = self.objects[chosen_label][ref_image_ind]

            yield self.preprocessing(image).to(self.device), self.preprocessing(ref_image).to(self.device), target

    def compose_iterations_raw(self):
        keys = list(self.objects.keys())
        while True:
            image_ind = np.random.randint(0, len(self.images_paths_set))
            image_path = self.images_paths_set[image_ind]
            image = Image.open(image_path)
            annot = self.annot_set[image_ind]

            labels = np.array([i[0] for i in annot])
            chosen_label = np.random.choice(np.unique(labels))
            bboxes = torch.tensor([np.array([i[1][0], i[1][1], i[2][0], i[2][1]])
                                  for i in annot], dtype=torch.float32)[labels == chosen_label]
            d = {}
            d['boxes'] = bboxes
            d['labels'] = torch.tensor([keys.index(i) for i in labels], dtype=torch.int64)
            target = [d]
            ref_image_ind = np.random.randint(0, len(self.objects[chosen_label]))
            ref_image = self.objects[chosen_label][ref_image_ind]

            yield image, ref_image, target

    def show(self, amount=1):
        i = iter(self.compose_iterations_raw())
        for _ in range(amount):
            image, ref_image, target = next(i)
            image = np.array(image)

            for xmin, ymin, xmax, ymax in target[0]['boxes']:
                image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)

            fg, ax = pyplot.subplots(1, 2)
            ax[0].imshow(image)
            ax[1].imshow(ref_image)
            pyplot.show()

    def __iter__(self):
        return iter(self.compose_iterations())


def collate_fn_cstm(coll):
    """
    Функция для правильного формирования батчей по данным SiameseDataset
    """
    input1_batched = list(c[0] for c in coll)
    input2_batched = list(c[1] for c in coll)
    output_batched = []
    for c in coll:
        output_batched += c[2]
    return input1_batched, input2_batched, output_batched


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


class BaseTrainProcess:
    """
    Модернизированный класс для обучения из семинара 7
    Добавлено логирование WandB
    """
    def __init__(self, hyp, device):
        start_time = strftime("%Y-%m-%d %H-%M-%S", gmtime())
        log_dir = (Path("logs") / start_time).as_posix()
        print('Log dir:', log_dir)
        self.writer = SummaryWriter(log_dir)
        self.logger = wandb

        self.best_loss = 1e100
        self.best_acc = 0.0
        self.current_epoch = -1
        self.device = device

        self.hyp = hyp

        self.lr_scheduler = None
        self.model = None
        self.optimizer = None
        self.criterion = None

        self.train_loader = None
        self.valid_loader = None

        self.init_params()

    def _init_data(self):

        train_coll, val_coll = collect_data()

        train_dataset = SiameseDataset(train_coll, self.device)
        valid_dataset = SiameseDataset(val_coll, self.device)

        train_dataset.show(5)
        valid_dataset.show()

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.hyp['batch_train'],
                                       num_workers=0,
                                       collate_fn=collate_fn_cstm
                                       )

        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.hyp['batch_val'],
                                       num_workers=0,
                                       collate_fn=collate_fn_cstm
                                       )

    def _init_model(self):
        self.model = SiameseModel()
        self.model.to(self.device)

        # self.optimizer = LARS(model_params, lr=0.2, weight_decay=1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hyp['lr'],
                                           weight_decay=self.hyp['weight_decay'])

        # "decay the learning rate with the cosine decay schedule without restarts"
        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: (epoch + 1) / 10.0)
        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            500,
            eta_min=0.05,
            last_epoch=-1,
        )

        self.criterion = nn.BCELoss().to(self.device)

    def _init_metrics(self):
        self.metrics = [
            Accuracy(task='binary'),
            F1Score(task='binary')
        ]
        self.metrics[0].__name__ = 'acc'
        self.metrics[1].__name__ = 'f1'

    def init_params(self):
        self._init_data()
        self._init_model()
        self._init_metrics()

    def save_checkpoint(self, loss_valid, path):
        if loss_valid <= self.best_loss:
            self.best_loss = loss_valid
            self.save_model(path)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.mainscheduler.state_dict()
        }, path)

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()

        cum_loss = 0.0
        proc_loss = 0.0
        size = self.hyp['N_train_batches_per_epoch']
        imagegen = iter(self.train_loader)
        pbar = tqdm(enumerate(np.arange(size)), total=size,
                    desc=f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}')
        loss = None
        for idx, _ in pbar:
            coll = next(imagegen)

            with torch.set_grad_enabled(True):
                loss_coll, _ = self.model(*coll)

                loss_coll['loss_rpn_box_reg'].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()

            cur_loss = loss_coll['loss_rpn_box_reg'].detach().cpu().numpy()
            cum_loss += cur_loss

            if loss is None:
                loss = {i: j.detach().cpu().numpy() for i, j in loss_coll.items()}
            else:
                loss = {i: loss[i] + loss_coll[i].detach().cpu().numpy() for i in loss.keys()}

            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            s = f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}'
            pbar.set_description(s)

        return {i: j / size / self.hyp['batch_train'] for i, j in loss.items()}

    def valid_step(self):
        # self.model.eval()

        cum_loss = 0.0
        proc_loss = 0.0
        size = self.hyp['N_val_batches_per_epoch']
        imagegen = iter(self.valid_loader)
        pbar = tqdm(enumerate(np.arange(size)), total=size,
                    desc=f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}')

        loss = None

        for idx, _ in pbar:
            coll = next(imagegen)

            with torch.set_grad_enabled(False):
                loss_coll, _ = self.model(*coll)

            cur_loss = loss_coll['loss_rpn_box_reg'].detach().cpu().numpy()
            cum_loss += cur_loss

            if loss is None:
                loss = {i: j.detach().cpu().numpy() for i, j in loss_coll.items()}
            else:
                loss = {i: loss[i] + loss_coll[i].detach().cpu().numpy() for i in loss.keys()}

            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            s = f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}'
            pbar.set_description(s)

        return {i: j / size / self.hyp['batch_val'] for i, j in loss.items()}

    def tst(self):
        i = iter(self.train_loader)
        coll = next(i)
        losses, _ = self.model.forward(*coll)
        print(losses)

    def run(self):
        best_w_path = 'best_V1.pt'
        last_w_path = 'last_v1.pt'

        train_losses = []
        valid_losses = []

        self.logger.init(project="Test_task", name=f"baseline")

        for epoch in range(self.hyp['epochs']):
            self.current_epoch = epoch

            train_coll = self.train_step()

            if epoch < 10:
                self.warmupscheduler.step()
            else:
                self.mainscheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            val_coll = self.valid_step()

            self.save_checkpoint(val_coll['loss_rpn_box_reg'], best_w_path)

            self.logger.log({'train': train_coll, 'val': val_coll}, step=epoch)

        self.save_model(last_w_path)
        torch.cuda.empty_cache()
        self.writer.close()
        self.logger.finish()

        return train_losses, valid_losses


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_siamese():
    device = 'cuda'
    hyps = {
        'batch_train': 2,
        'batch_val': 4,
        'N_train_batches_per_epoch': 20,
        'N_val_batches_per_epoch': 10,
        'lr': 1e-4,
        'lrf': 0.02,
        'epochs': 30,
        'weight_decay': 0.0004,
        'seed': 42
    }

    set_seed(hyps['seed'])
    trainer = BaseTrainProcess(hyps, device)
    # trainer.tst()
    train_losses, valid_losses = trainer.run()
    # print(train_losses)
    # print(valid_losses)


if __name__ == '__main__':
    train_siamese()
