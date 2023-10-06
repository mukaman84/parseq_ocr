#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from omegaconf import DictConfig, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig
import shutil
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize
import os

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from pytorch_lightning import Trainer, LightningModule

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import create_model, load_pre_trained_mode_with_zero_epoch
from strhub.models.utils import load_from_checkpoint, parse_model_args

# log = logging.getLogger(__name__)
import logging

class TuneReportCheckpointPruneCallback(TuneReportCheckpointCallback):

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        self._checkpoint._handle(trainer, pl_module)
        # Prune older checkpoints
        for old in sorted(Path(tune.get_trial_dir()).glob('checkpoint_epoch=*-step=*'), key=os.path.getmtime)[:-1]:
            log.info(f'Deleting old checkpoint: {old}')
            shutil.rmtree(old)
        self._report._handle(trainer, pl_module)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = None
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        gpus = config.trainer.get('gpus', 0)
        if gpus:
            # Use mixed-precision trainingd
            config.trainer.precision = 16
        # if gpus > 1:
        #     # Use DDP
        #     config.trainer.strategy = 'ddp'
        #     # DDP optimizations
        #     trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
        #     # Scale steps-based config
        #     # config.trainer.val_check_interval //= gpus
        #     # if config.trainer.get('max_steps', -1) > 0:
        #     #     config.trainer.max_steps //= gpus

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    # If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        model: BaseSystem = create_model(config.pretrained, True)

    else:
        model: BaseSystem = hydra.utils.instantiate(config.model)
        # model = model.load_from_checkpoint(config.ckpt_path)
        # model: BaseSystem = load_pre_trained_mode_with_zero_epoch(config.ckpt_path)
        # checkpoint = torch.load(config.ckpt_path)
        # print("model_state_dict : ", checkpoint.keys())

        # ModelClass = _get_model_class(config.ckpt_path)
        # model = ModelClass.load_from_checkpoint(checkpoint_path)

        # model = model.load_state_dict(checkpoint['state_dict'])

    print(summarize(model, max_depth=1))# if model.hparams.name.startswith('parseq') else 2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(monitor='val_accuracy', mode='max', save_on_train_epoch_end =True, save_top_k=10, save_last=True,
                                 filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}')
    swa = StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=0.75)
    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else \
        str(Path(config.ckpt_path).parents[1].absolute())



    print("config.trainer.max_steps",config.trainer.max_steps)

    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(cwd, '', '.'),
                                               strategy=trainer_strategy, enable_model_summary=True,
                                               callbacks=[checkpoint, swa], max_steps=39274000)

    # checkpoint = torch.load(config.ckpt_path)

    # model.load_state_dict(checkpoint['state_dict'])

    trainer.fit(model, datamodule=datamodule)#, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
