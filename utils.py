import datetime
import dgl
import errno
import numpy as np
import os
from pathlib import Path
import pickle
import random
import torch

from pprint import pprint
from scipy import sparse
from scipy import io as sio

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

# The configuration 
default_configure = {       
    'hidden_units': 64
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['device'] = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    return args

class EarlyStopping(object):
    def __init__(self, patience=10, is_homo=True):
        dt = datetime.datetime.now()
        filedirectory = 'early_stop_model/'
        Path(filedirectory).mkdir(parents=True, exist_ok=True)
        self.filename = filedirectory + 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
        self.renorm_probs = None
        self.test_acc = None
        self.is_homo = is_homo

    def step(self, loss, acc, model, renorm_probs=None, test_acc=None):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
            self.renorm_probs = renorm_probs
            self.test_acc = test_acc
        elif (not self.is_homo and (loss > self.best_loss) and (acc < self.best_acc)) or \
             (self.is_homo and (acc < self.best_acc)):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (not self.is_homo and (loss <= self.best_loss) and (acc >= self.best_acc)) or \
               (self.is_homo and acc >= self.best_acc):
                self.save_checkpoint(model)
                self.renorm_probs = renorm_probs
                self.test_acc = test_acc
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
