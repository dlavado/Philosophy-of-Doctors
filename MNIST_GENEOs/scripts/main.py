

import ast
from datetime import datetime
from typing import List
import torch
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
import pytorch_lightning.callbacks as  pl_callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

import sys
import os
# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.models.cnn import Lit_CNN_Classifier
from core.datasets.mnist import MNISTDataModule

from core.models.lit_wrapper import callback_model_checkpoint
from utils import utils


def init_callbacks(ckpt_dir):
    # Call back definition
    callbacks = []
    model_ckpts: List[pl_callbacks.ModelCheckpoint] = []

    ckpt_metrics = [str(met) for met in utils.init_metrics()]

    for metric in ckpt_metrics:
        model_ckpts.append(
            callback_model_checkpoint(
                dirpath=ckpt_dir,
                filename=f"{metric}",
                monitor=f"train_{metric}",
                mode="max",
                save_top_k=2,
                save_last=False,
                every_n_epochs=wandb.config.checkpoint_every_n_epochs,
                every_n_train_steps=wandb.config.checkpoint_every_n_steps,
                verbose=False,
            )
        )


    model_ckpts.append( # train loss checkpoint
        callback_model_checkpoint(
            dirpath=ckpt_dir, #None for default logger dir
            filename=f"train_loss",
            monitor=f"train_loss",
            mode="min",
            every_n_epochs=wandb.config.checkpoint_every_n_epochs,
            every_n_train_steps=wandb.config.checkpoint_every_n_steps,
            verbose=False,
        )
    )

    callbacks.extend(model_ckpts)

    early_stop_callback = EarlyStopping(monitor=wandb.config.early_stop_metric, 
                                        min_delta=0.00, 
                                        patience=30, 
                                        verbose=False, 
                                        mode="max")

    callbacks.append(early_stop_callback)

    return callbacks

def init_model(model_wrapper_class=Lit_CNN_Classifier, ckpt_path=None):

    if wandb.config.resume_from_checkpoint:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        
        print(f"Resuming from checkpoint {ckpt_path}")
        model = model_wrapper_class.load_from_checkpoint(ckpt_path,
                                                        optimizer=wandb.config.optimizer,
                                                        learning_rate=wandb.config.learning_rate,
                                                        metric_initilizer=utils.init_metrics)
        
    else:
        model = model_wrapper_class(
            in_channels=wandb.config.in_channels,
            hidden_dim=wandb.config.hidden_dim,
            kernel_size=wandb.config.kernel_size,
            optimizer_name=wandb.config.optimizer,
            learning_rate=wandb.config.learning_rate,
            metric_initilizer=utils.init_metrics,
        )
    return model


def init_data(data_path):

    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    
    mnist = MNISTDataModule(data_dir=data_path,
                            transform=transform)
    return mnist



def main():

    # INIT CALLBACKS
    # --------------
  
    if not wandb.config.resume_from_checkpoint:
        ckpt_dir = os.path.join(wandb.run.dir, "checkpoints") 
    else:
        ckpt_dir = wandb.config.checkpoint_dir

    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')

    callbacks = init_callbacks(ckpt_dir)


    # INIT MODEL
    # ----------

    model = init_model(Lit_CNN_Classifier, ckpt_path)


    # INIT DATA
    # ---------

    data_path = wandb.config.data_path
    mnist = init_data(data_path)


    # INIT TRAINER
    # ------------

    # WandbLogger
    wandb_logger = WandbLogger(project=f"{project_name}",
                               log_model=True, 
                               name=wandb.run.name, 
                               config=wandb.config)
    
    wandb_logger.watch(model, log="all", log_freq=100)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        detect_anomaly=True,
        max_epochs=wandb.config.max_epochs,
        gpus=wandb.config.gpus,
        #fast_dev_run = wandb.config.fast_dev_run,
        auto_lr_find=wandb.config.auto_lr_find,
        auto_scale_batch_size=wandb.config.auto_scale_batch_size,
        profiler=wandb.config.profiler,
        enable_model_summary=True,
        accumulate_grad_batches = ast.literal_eval(wandb.config.accumulate_grad_batches),
        #resume_from_checkpoint=ckpt_path
    )

    if wandb.config.auto_lr_find or wandb.config.auto_scale_batch_size:
        trainer.tune(model, mnist) # auto_lr_find and auto_scale_batch_size
        print(f"Learning rate in use is: {model.hparams.learning_rate}")


    trainer.fit(model, mnist)

    print(f"{'='*20} Model ckpt scores {'='*20}")

    for ckpt in trainer.callbacks:
        if isinstance(ckpt, pl_callbacks.ModelCheckpoint):
            print(f"{ckpt.monitor} checkpoint : score {ckpt.best_model_score}")

    wandb_logger.experiment.unwatch(model)

    # 6 TEST
    # ------

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} does not exist. Using last checkpoint.")
        ckpt_path = None

    if wandb.config.save_onnx:
        print("Saving ONNX model...")
        onnx_file_path = os.path.join(ckpt_dir, f"{project_name}.onnx")
        input_sample = next(iter(mnist.test_dataloader()))
        model.to_onnx(onnx_file_path, input_sample, export_params=True)
        wandb_logger.log({"onnx_model": wandb.File(onnx_file_path)})

    trainer.test(model, 
                 datamodule=mnist,
                 ckpt_path=ckpt_path) # use the last checkpoint
    






if __name__ == '__main__':

    model_name = 'cnn'
    dataset_name = 'mnist'
    project_name = f"{model_name}_{dataset_name}"

    run_name = f"{project_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    main()