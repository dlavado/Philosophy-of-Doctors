

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, MetricCollection, Precision, Recall



def main_arg_parser():
    parser = argparse.ArgumentParser(description="Process script arguments")

    parser.add_argument('--wandb_sweep', action='store_true', default=None, help='If True, the script is run by wandb sweep')

    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use')
    parser.add_argument('--model', type=str, default='cnn', help='Model to use')

    return parser



def init_metrics(num_classes=10):
    return MetricCollection([
        Accuracy(num_classes=num_classes, multiclass=True),
        Precision(num_classes=num_classes, multiclass=True),
        Recall(num_classes=num_classes, multiclass=True)
    ])




def view_classify(img, logits):
    ''' 
    Function for viewing an image and it's predicted classes.
    '''

    img = img.view(1, 28, 28)
    ps = torch.exp(logits)
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()