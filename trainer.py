import os
from decimal import Decimal

import imageio
import numpy as np
import torch
from tqdm import tqdm

import utility
from data import common


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_models = self.model.dual_models
        self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
            sr = self.model(lr[0])
            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

            # compute primary loss
            loss_primary = self.loss(sr[-1], hr)
            for i in range(1, len(sr)):
                loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

            # compute dual loss
            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])

            # compute total loss
            loss = loss_primary + self.opt.dual_weight * loss_dual

            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    @torch.no_grad()
    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        scale = max(self.scale)
        for si, s in enumerate([scale]):
            eval_psnr = 0
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for _, (lr, hr, filename) in enumerate(tqdm_test):
                filename = filename[0]
                # no_eval = (hr.nelement() == 1)
                no_eval = True
                if not no_eval:
                    lr, hr = self.prepare(lr, hr)
                else:
                    lr, = self.prepare(lr)

                sr = self.model(lr[0])
                if isinstance(sr, list):
                    sr = sr[-1]

                sr = utility.quantize(sr, self.opt.rgb_range)

                if not no_eval:
                    eval_psnr += utility.calc_psnr(
                        sr, hr, s, self.opt.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )

                # save test results

                if self.opt.save_results:
                    self.ckp.save_results_nopostfix(filename, sr, s)

            # self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                    self.opt.data_test, s,
                    self.ckp.log[-1, si],
                    best[0][si],
                    best[1][si] + 1
                )
            )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    @torch.no_grad()
    def transform(self, file_path):
        filename, format = os.path.splitext(os.path.basename(file_path))
        self.model.eval()
        lr_array = imageio.imread(file_path)
        lr, _ = common.set_channel([lr_array], lr_array, n_channels=3)
        lr_tensor, _ = common.np2Tensor(lr, lr[0], rgb_range=255)
        scale = max(self.scale)
        lr_tensor = self.prepare(lr_tensor)
        lr_input = torch.stack(lr_tensor[0])
        sr = self.model(lr_input)
        if isinstance(sr, list):
            sr = sr[-1]
        sr = utility.quantize(sr, self.opt.rgb_range)
        # save test results
        self.ckp.save_results_nopostfix(filename, sr, scale)

    @torch.no_grad()
    def transform_frame(self, lr_array):
        lr, _ = common.set_channel([lr_array], lr_array, n_channels=3)
        lr_tensor, _ = common.np2Tensor(lr, lr[0], rgb_range=255)
        lr_tensor = self.prepare(lr_tensor)
        lr_input = torch.stack(lr_tensor[0])
        sr = self.model(lr_input)

        sr_array2 = utility.quantize(sr[1], self.opt.rgb_range)
        frame_SR2 = sr_array2.permute(0, 2, 3, 1)  # 转换维度，把颜色维度放在最后
        frame_SR2 = np.squeeze(frame_SR2, 0).cpu()
        frame_SR2 = np.array(frame_SR2)

        sr_array4 = utility.quantize(sr[2], self.opt.rgb_range)
        frame_SR4 = sr_array4.permute(0, 2, 3, 1)  # 转换维度，把颜色维度放在最后
        frame_SR4 = np.squeeze(frame_SR4, 0).cpu()
        frame_SR4 = np.array(frame_SR4)

        return frame_SR2, frame_SR4

    @torch.no_grad()
    def transform_picture(self, file_path):
        self.model.eval()
        lr_array = imageio.imread(file_path)
        lr, _ = common.set_channel([lr_array], lr_array, n_channels=3)
        lr_tensor, _ = common.np2Tensor(lr, lr[0], rgb_range=255)
        lr_tensor = self.prepare(lr_tensor)
        lr_input = torch.stack(lr_tensor[0])
        sr = self.model(lr_input)

        sr_array2 = utility.quantize(sr[1], self.opt.rgb_range)
        image_SR2 = sr_array2.permute(0, 2, 3, 1)  # 转换维度，把颜色维度放在最后
        image_SR2 = np.squeeze(image_SR2, 0).cpu()
        image_SR2 = np.array(image_SR2)

        sr_array4 = utility.quantize(sr[2], self.opt.rgb_range)
        image_SR4 = sr_array4.permute(0, 2, 3, 1)  # 转换维度，把颜色维度放在最后
        image_SR4 = np.squeeze(image_SR4, 0).cpu()
        image_SR4 = np.array(image_SR4)

        return image_SR2, image_SR4

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args) > 1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]],

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
