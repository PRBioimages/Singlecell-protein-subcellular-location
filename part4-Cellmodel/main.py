from utils import parse_args, prepare_for_result
from dataloaders import get_dataloader
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from basic_train import basic_train
from scheduler import get_scheduler
from utils import load_matched_state
from torch.utils.tensorboard import SummaryWriter
import torch
try:
    from apex import amp
except:
    pass
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    print('[ √ ] Starting training')
    args, cfg = parse_args()
    # print(cfg)
    result_path = prepare_for_result(cfg)
    writer = SummaryWriter(log_dir=result_path)
    cfg.dump_json(result_path / 'config.json')

    # training model

    train_dl, _, _ = get_dataloader(cfg)(cfg).get_dataloader()
    print('[ i ] The length of train_dl is {}'.format(len(train_dl)))
    model = get_model(cfg).cuda()
    if not cfg.model.from_checkpoint == 'none':
        print('[ ! ] loading model from checkpoint: {}'.format(cfg.model.from_checkpoint))
        load_matched_state(model, torch.load(cfg.model.from_checkpoint, map_location='cpu'))
        # model.load_state_dict(torch.load(cfg.model.from_checkpoint))
    if cfg.loss.name == 'weighted_ce_loss':
        # if we use weighted ce loss, we load the loss here.
        weights = torch.Tensor(cfg.loss.param['weight']).cuda()
        loss_func = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
    else:
        loss_func = get_loss(cfg)
    if cfg.train.freeze_backbond:
        print('[ i ] freeze backbone')
        model.model.requires_grad = False
    optimizer = get_optimizer(model, cfg)
    print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name, cfg.optimizer.name))
    if not cfg.scheduler.name == 'none':
        scheduler = get_scheduler(cfg, optimizer, len(train_dl))
    else:
        scheduler = None
    if len(cfg.basic.GPU) > 1:
        model = torch.nn.DataParallel(model)

    basic_train(cfg, model, train_dl, _, loss_func, optimizer, result_path, scheduler, writer)
