from path import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataloaders.datasets import *
from dataloaders.transform_loader import get_tfms
import os
from dataloaders.datasets import a_ordinary_collect_method
from utils import Config


class RandomKTrainTestSplit:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pseodu = None
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        if cfg.experiment.file == 'none':
            csv_file = 'all_cell_pseudo.csv'
        else:
            csv_file = cfg.experiment.file
        train = pd.concat([pd.read_csv(path / 'meta_csv' / cf) for cf in csv_file.split(';')], axis=0)

        # self.train_meta, self.valid_meta = (train[train.fold != cfg.experiment.run_fold],
        #                                     train[train.fold == cfg.experiment.run_fold])
        self.train_meta = train[train.fold != cfg.experiment.run_fold]
        # self.train_meta = self.train_meta.iloc[500:1000, :].reset_index(drop=True)
        # This part does not participate in the training and only tests whether there is a changeing trend of related metrics similar to the single-cell results.
        # self.valid_meta = self.valid_meta.iloc[:500, :].reset_index(drop=True)

        # print(train.head())
        if cfg.basic.debug:
            print('[ W ] Debug Mode!, down sample')
            self.train_meta = self.train_meta.sample(frac=0.00005)
            # self.valid_meta = self.valid_meta.sample(frac=1)

    def get_dataloader(self, train_shuffle=True, tta=-1, tta_tfms=None):
        print('[ √ ] Using augmentation: {} & {}, image size: {}'.format(
            self.cfg.transform.name, self.cfg.transform.val_name, self.cfg.transform.size
        ))
        if self.cfg.transform.name == 'None':
            train_tfms = None
        else:
            train_tfms = get_tfms(self.cfg.transform.name)
        if tta_tfms:
            val_tfms = tta_tfms
        elif self.cfg.transform.val_name == 'None':
            val_tfms = None
        else:
            val_tfms = get_tfms(self.cfg.transform.val_name)
        train_ds = STRDataset(self.train_meta, train_tfms, size=self.cfg.transform.size,
                              cfg=self.cfg, mode='train')

        train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size, prefetch_factor=4,
                                  num_workers=self.cfg.transform.num_preprocessor,
                                  shuffle=train_shuffle, drop_last=True, pin_memory=False,
                                  persistent_workers=True)

        # valid_ds = STRDataset(df=self.valid_meta, tfms=val_tfms, size=self.cfg.transform.size,
        #            cfg=self.cfg, mode='valid')
        # valid_dl = DataLoader(dataset=valid_ds, batch_size=self.cfg.eval.batch_size, drop_last=True,
        #                       num_workers=self.cfg.transform.num_preprocessor, pin_memory=False)
        return train_dl, None, None
