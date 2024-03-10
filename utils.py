import os.path as osp

from lightning.pytorch import LightningModule, Trainer, callbacks
from diffusers.configuration_utils import ConfigMixin

from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from diffusers.configuration_utils import FrozenDict


class PipelineCheckpoint(callbacks.ModelCheckpoint):

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint) -> None:
        # only ema parameters (if any) saved in pipeline
        with pl_module.maybe_ema():
            pipe_path = osp.join(
                osp.dirname(self.best_model_path),
                f'pipeline-{pl_module.current_epoch}'
            )
            pl_module.save_pretrained(pipe_path)

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)


def _fix_hydra_config_serialization(conf_mixin: ConfigMixin):
    # This is a hack due to incompatibility between hydra and diffusers
    new_internal_dict = {}
    for k, v in conf_mixin._internal_dict.items():
        if isinstance(v, ListConfig):
            new_internal_dict[k] = list(v)
        elif isinstance(v, DictConfig):
            new_internal_dict[k] = dict(v)
        else:
            new_internal_dict[k] = v
    conf_mixin._internal_dict = FrozenDict(new_internal_dict)