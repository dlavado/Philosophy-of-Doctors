

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import pytorch_lightning as pl
import torchvision
import pytorch_lightning.loggers as pl_loggers




def init_cifar10dm(data_dir: str = "./", batch_size: int = 32, num_workers: int = 12):

    train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

    test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

    return CIFAR10DataModule(data_dir=data_dir, 
                            batch_size=batch_size, 
                            num_workers=num_workers,
                            train_transforms=train_transforms,
                            test_transforms=test_transforms,
                            val_transforms=test_transforms,
                            )


