import configparser
import torch
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
from tqdm import tqdm
from ast import literal_eval
from math import ceil

from src.networks.network_factory import build_network
from src.dataset.dataloader_factory import build_dataloaders
from src.training_utils.optimizer_factory import build_optimizer
from src.training_utils.loss_factory import build_losses


class Trainer:
    def __init__(self, config):
        """

        :param config:          training config object
        """
        self.config = config

        self.input_shape = literal_eval(config['default']['input_shape'])
        self.downscale = config['default'].getint('downscale')
        self.paf_loss_multiplier = config['loss'].getfloat('paf_loss_multiplier')
        self.kp_loss_multiplier = config['loss'].getfloat('class_loss_multiplier')
        self.deep_supervion_decay = config['loss'].getfloat('deep_supervison_decay')
        self.n_pafs = config['network'].getint('n_pafs')

        self.model = build_network(config).cuda()
        if config['network'].getboolean('load'):
            self.model.load_state_dict(torch.load(config['network']['path']))

        self.optimizer = build_optimizer(self.model.parameters(), config)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=1,
                                                                    min_lr=5e-5, verbose=True)

        self.paf_loss, self.kp_loss = build_losses(config)

        self.train_dataloader, self.val_dataloader = build_dataloaders(config)

        self.train_steps = ceil(len(self.train_dataloader.dataset) / self.config['default'].getint('batch_size'))
        self.val_steps = ceil(len(self.val_dataloader.dataset) / self.config['default'].getint('batch_size'))

        self.writer = SummaryWriter(log_dir=config['default']['log_path'])

    def get_loss(self, pafs, kps, o1, o2, n_pafs, n_kps):
        """Calculates loss for a batch

        :param pafs:            gt pafs
        :param kps:             gt keypoints
        :param o1:              pred pafs
        :param o2:              pred keypoints
        :param n_pafs:          number of pafs for each paf type
        :param n_kps:           number of keypoints for each keypoint type
        :return:
        """
        n_pafs = n_pafs.repeat_interleave(2, dim=1)

        loss_paf = sum([sum([sum(self.deep_supervion_decay**i * self.paf_loss_multiplier * (n_pafs[:, n].float() + 0.1) *
                                 self.paf_loss(o[:, n, :, :].float(), pafs[:, n, :, :].float())) for n in range(n_pafs.shape[1])]) for i, o in enumerate(o1[::-1])])

        loss_kp = sum([sum([sum(self.deep_supervion_decay**i * self.kp_loss_multiplier * (n_kps[:, n].float() + 0.1) *
                                self.kp_loss(o[:, n, :, :].float(), kps[:, n, :, :].float())) for n in range(n_kps.shape[1])]) for i, o in enumerate(o2[::-1])])

        loss_total = loss_paf + loss_kp

        return loss_paf, loss_kp, loss_total

    def train(self):
        """Training and validation loops

        """
        self.model.train()
        global_step_c = 0
        for epoch in range(config['default'].getint('epochs')):

            # TRAINING LOOP
            train_loss, train_paf_loss, train_kp_loss = [], [], []
            for imgs, pafs, kps, n_pafs, n_kps in tqdm(self.train_dataloader, total=self.train_steps, desc=f'[TRAIN] Epoch {epoch}'):
                imgs, pafs, kps, n_pafs, n_kps = imgs.cuda(), pafs.cuda(), kps.cuda(), n_pafs.cuda(), n_kps.cuda()

                self.optimizer.zero_grad()

                o1, o2 = self.model(imgs.float())

                loss_paf, loss_kp, loss_total = self.get_loss(pafs, kps, o1, o2, n_pafs, n_kps)

                loss_total.backward()

                train_paf_loss.append(loss_paf.item())
                train_kp_loss.append(loss_kp.item())
                train_loss.append(loss_total.item())

                self.optimizer.step()

                if not (global_step_c % 1000):
                    img = cv2.cvtColor(imgs[0].cpu().detach().numpy().transpose((1, 2, 0)).astype(np.float32), cv2.COLOR_BGR2RGB)

                    paf_gt_x = np.max(np.abs(cv2.resize(pafs[0].cpu().detach().numpy().transpose((1, 2, 0))[:, :, [i for i in range(0, self.n_pafs * 2, 2)]],
                                                        (self.input_shape[1], self.input_shape[0]))), axis=-1)
                    paf_gt_y = np.max(np.abs(cv2.resize(pafs[0].cpu().detach().numpy().transpose((1, 2, 0))[:, :, [i for i in range(1, self.n_pafs * 2, 2)]],
                                                        (self.input_shape[1], self.input_shape[0]))), axis=-1)

                    paf_pred_x = np.max(np.abs(cv2.resize(o1[self.config['network'].getint('hourglass_stages') - 1][0]
                                                          .cpu().detach().numpy().transpose((1, 2, 0))[:, :, [i for i in range(0, self.n_pafs * 2, 2)]],
                                                          (self.input_shape[1], self.input_shape[0]))), axis=-1)
                    paf_pred_y = np.max(np.abs(cv2.resize(o1[self.config['network'].getint('hourglass_stages') - 1][0]
                                                          .cpu().detach().numpy().transpose((1, 2, 0))[:, :, [i for i in range(1, self.n_pafs * 2, 2)]],
                                                          (self.input_shape[1], self.input_shape[0]))), axis=-1)

                    kp_gt = np.max(cv2.resize(kps[0].cpu().detach().numpy().transpose((1, 2, 0)),
                                              (self.input_shape[1], self.input_shape[0])), axis=2)
                    kp_pred = np.max(cv2.resize(o2[self.config['network'].getint('hourglass_stages') - 1][0].cpu().detach().numpy().transpose((1, 2, 0)),
                                                (self.input_shape[1], self.input_shape[0])), axis=2)

                    self.writer.add_image('Train/image', img, global_step_c, dataformats='HWC')
                    self.writer.add_image('Train/paf_gt_x', paf_gt_x, global_step_c, dataformats='HW')
                    self.writer.add_image('Train/paf_gt_y', paf_gt_y, global_step_c, dataformats='HW')
                    self.writer.add_image('Train/paf_pred_x', paf_pred_x, global_step_c, dataformats='HW')
                    self.writer.add_image('Train/paf_pred_y', paf_pred_y, global_step_c, dataformats='HW')
                    self.writer.add_image('Train/kp_gt', kp_gt, global_step_c, dataformats='HW')
                    self.writer.add_image('Train/kp_pred', kp_pred, global_step_c, dataformats='HW')

                global_step_c += 1

            print('\n[TRAIN] Epoch {0: <4}: {1: <6}\n'.format(epoch, np.round(np.mean(train_loss), 4)))

            self.writer.add_scalar('Loss/train/epoch_paf', np.mean(train_paf_loss), epoch)
            self.writer.add_scalar('Loss/train/epoch_kp', np.mean(train_kp_loss), epoch)
            self.writer.add_scalar('Loss/train/epoch_loss', np.mean(train_loss), epoch)

            # VALIDATION LOOP
            val_loss, val_paf_loss, val_kp_loss = [], [], []
            self.model.eval()
            with torch.no_grad():
                for imgs, pafs, kps, n_pafs, n_kps in tqdm(self.val_dataloader, total=self.val_steps, desc=f'[VAL] Epoch {epoch}'):
                    imgs, pafs, kps, n_pafs, n_kps = imgs.cuda(), pafs.cuda(), kps.cuda(), n_pafs.cuda(), n_kps.cuda()

                    o1, o2 = self.model(imgs.float())

                    loss_paf, loss_kp, loss_total = self.get_loss(pafs, kps, o1, o2, n_pafs, n_kps)

                    val_paf_loss.append(loss_paf.item())
                    val_kp_loss.append(loss_kp.item())
                    val_loss.append(loss_total.item())

                print('\n[VAL] Epoch {0: <4}: {1: <6}\n'.format(epoch, np.round(np.mean(val_loss), 4)))

                self.writer.add_scalar('Loss/val/epoch_paf', np.mean(val_paf_loss), epoch)
                self.writer.add_scalar('Loss/val/epoch_kp', np.mean(val_kp_loss), epoch)
                self.writer.add_scalar('Loss/val/epoch_loss', np.mean(val_loss), epoch)

                torch.save(self.model.state_dict(), f'../saved/models/{self.config["network"]["name"]}_{epoch}.pth')

                self.scheduler.step(np.mean(val_loss))

        print('Finished Training')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./train.cfg')

    trainer = Trainer(config)
    trainer.train()
