from .modelsLinks.BIT.models.evaluator import CDEvaluator as Bit
from .modelsLinks.ChangeFormer.models.evaluator import CDEvaluator as ChangeFormer

from .methods import Resize, Crop, SlidingWindowAverage, GaussianSlidingWindow
from .helpers.gaussian_helpers import make_generator
from .helpers.constants import AttributeDict, CHANGEFORMER, BIT, CHANGEREX, CHANGE_FORMER_CONFIGS, BIT_CONFIGS, CHANGEREX_CONFIGS, \
    RESIZE, CROP, SLIDINGWINDOWAVERAGE, GAUSSIANSLIDINGWINDOW, DEVICE

from .helpers.metrics import Metrics

# from modelsLinks.BIT.models.evaluator import CDEvaluator as Bit
# from methods import Resize, Crop, SlidingWindowAverage, GaussianSlidingWindow
# from helpers.gaussian_helpers import make_generator
# from helpers.constants import AttributeDict, CHANGEFORMER, BIT, CHANGEREX, CHANGE_FORMER_CONFIGS, BIT_CONFIGS, CHANGEREX_CONFIGS, \
#      RESIZE, CROP, SLIDINGWINDOWAVERAGE, GAUSSIANSLIDINGWINDOW, DEVICE
#
# from helpers.metrics import Metrics

import os
import numpy as np
import json
import cv2
import subprocess

import torch

def get_device(configs):
    str_ids = configs.gpu_ids.split(',') 
    configs.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            configs.gpu_ids.append(id)
    if len(configs.gpu_ids) > 0:
        torch.cuda.set_device(configs.gpu_ids[0])

class Middleware:
    def __init__(self, config):
        self.config = config 
        self.img_shape = (self.config.batch_size, 2) + tuple(self.config.img_shape[self.config.data_name])
        self.metric_calculator = Metrics(self.config.METRICS)
        self.model_mapper = {
            CHANGEFORMER: self.__predictChangeFormer__,
            BIT: self.__predictBIT__,
            CHANGEREX: self.__predictChangerEx__ 
        }

        self.method_mapper = {
            RESIZE: Resize,
            CROP: Crop,
            SLIDINGWINDOWAVERAGE: SlidingWindowAverage,
            GAUSSIANSLIDINGWINDOW: GaussianSlidingWindow
        }


    def _save_results(self, method_name, model_name, logits, targets, image_names):
        if targets is not None:
            labels = targets.detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

        for i in range(preds.shape[0]):
            image_path = os.path.join(self.config.results_dir, model_name, method_name, self.config.data_name, image_names[i])
            if targets is None:
                cv2.imwrite(image_path, preds[i])
            else:
                cv2.imwrite(image_path, np.concatenate([255*preds[i], 123*np.ones((preds[i].shape[0], 5)), 255*labels[i][0]], axis=1))
        

    def __predictChangeFormer__(self, method_name, dataloader):
        args = open(CHANGE_FORMER_CONFIGS)
        args = json.load(args)
        configs = AttributeDict(args)

        get_device(configs)

        model = ChangeFormer(configs, dataloader)
        model.device = DEVICE

        model._load_checkpoint(configs.checkpoint_name)

        model.logger.write('Begin evaluation...\n')
        model._clear_cache()
        model.is_training = False
        model.net_G.eval()


        for batch_id, batch in enumerate(dataloader, 0): 
            with torch.no_grad():
                results = []
                inp = self.method.process(batch)
                batch_depth = inp[0][0].shape[0]
                if batch_depth != self.config.batch_size:
                    self.img_shape = (batch_depth, ) + self.img_shape[1:]
                for tupl in inp:
                    inp1 = tupl[0].to(DEVICE)
                    inp2 = tupl[1].to(DEVICE)
                    output = model.net_G(inp1, inp2)[-1]
                    results.append(output)
                gen = make_generator(results)
                pred = self.method.merge(gen, self.img_shape)
                self._save_results(method_name, CHANGEFORMER, pred, None, batch["name"])


    def __predictBIT__(self, method_name, dataloader):
        args = open(BIT_CONFIGS)
        args = json.load(args)
        configs = AttributeDict(args)

        get_device(configs)

        model = Bit(configs, dataloader)
        model.device = DEVICE

        model._load_checkpoint(configs.checkpoint_name)

        model.logger.write('Begin evaluation...\n')
        model._clear_cache()
        model.is_training = False
        model.net_G.eval()

        for batch_id, batch in enumerate(dataloader, 0):
            with torch.no_grad():
                results = []
                inp = self.method.process(batch)
                batch_depth = inp[0][0].shape[0]
                if batch_depth != self.config.batch_size:
                    self.img_shape = (batch_depth, ) + self.img_shape[1:]
                for tupl in inp:
                    inp1 = tupl[0].to(DEVICE)
                    inp2 = tupl[1].to(DEVICE)
                    results.append(model.net_G(inp1, inp2)[-1])
                gen = make_generator(results)
                pred = self.method.merge(gen, self.img_shape)
                if len(pred.shape) == 3:
                    pred = pred.unsqueeze(0)
                #targets = batch[self.config["label_dir"]]
                #self.metric_calculator._update_metrics_with_batch(pred, targets)
                self._save_results(method_name, BIT, pred, None, batch["name"])


    def __predictChangerEx__(self, method_name, dataloader):
        args = open(CHANGEREX_CONFIGS)
        args = json.load(args)
        args = AttributeDict(args)

        if 'LOCAL_RANK' not in os.environ: 
            os.environ['LOCAL_RANK'] = str(args['local_rank'])

        cfg = Config.fromfile(args.config)
        cfg.launcher = args.launcher

        if 'work-dir' in args:
            cfg.work_dir = args['work-dir']
        elif cfg.get('work_dir', None) is None:
            cfg.work_dir = os.path.join('./work_dirs',
                                        os.path.splitext(os.path.basename(args['config']))[0])

        cfg.load_from = args.checkpoint

        if args.tta:
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
            cfg.tta_model.module = cfg.model
            cfg.model = cfg.tta_model

        runner = Runner.from_cfg(cfg)
        runner._test_loop = runner.build_test_loop(runner._test_loop)
        runner.call_hook("before_run")
        runner.load_or_resume()
        runner.call_hook("before_test")
        runner.call_hook("before_test_epoch")
        runner.model.eval()

        def _make_data_samples(window_shape, ori_shape):
            img_meta = dict(img_shape=window_shape, ori_shape=ori_shape)
            data_sample = SegDataSample(metainfo=img_meta)
            gt_segmentations = PixelData()
            gt_segmentations.data = torch.zeros(window_shape)
            data_sample.gt_sem_seg = gt_segmentations
            return data_sample


        def _make_inputs(inp1, inp2):
            inp1 = inp1[[2, 1, 0], ...]
            inp2 = inp2[[2, 1, 0], ...]

            res = [torch.cat([inp1, inp2], dim=0)]
            return res
        

        def _segmap2tensor(out):
            res = out[0].seg_logits.data 
            return res

        window_shape = (self.config.window_size, self.config.window_size)
        data_sample = _make_data_samples(window_shape, window_shape)
        for batch_id, batch in enumerate(dataloader, 0): 
            with torch.no_grad():
                results = []
                inp = self.method.process(batch)
                batch_depth = inp[0][0].shape[0]
                if batch_depth != self.config.batch_size:
                    img_shape = (batch_depth, ) + img_shape[1:]
                for tupl in inp:
                    inp1 = tupl[0].to(DEVICE)
                    inp2 = tupl[1].to(DEVICE)
                    inp1 = ((inp1*0.5 + 0.5) * 255).to(torch.uint8)
                    inp2 = ((inp2*0.5 + 0.5) * 255).to(torch.uint8)

                    outs = []
                    for i in range(self.img_shape[0]):
                        inp = _make_inputs(inp1[i], inp2[i]) 
                        mini_batch = {"inputs": inp, "data_samples": [data_sample]}
                        out = runner.model.test_step(mini_batch)
                        tensor_out = _segmap2tensor(out).unsqueeze(0)
                        outs.append(tensor_out)
                    output = torch.cat(outs, dim=0)
                    results.append(output)
                gen = make_generator(results)
                pred = self.method.merge(gen, self.img_shape)
                targets = batch[self.config.label_dir]                
                #self.metric_calculator._update_metrics_with_batch(pred, targets)
                self._save_results(method_name, CHANGEREX, pred, targets, batch["name"])

    def predict(self, model_name, method_name, dataloader):
        self.method = self.method_mapper[method_name](window_size=self.config.window_size, stride=self.config.stride, sigma=self.config.sigma)
        self.model_mapper[model_name](method_name, dataloader)
        #self.metric_calculator._compute_all_metrics()

