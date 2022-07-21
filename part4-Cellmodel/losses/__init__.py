from losses.regular import ce, label_smooth_ce, label_smooth_ce_ohem, mse, mae, bce, sl1, bce_mse
from losses.regular import focal_loss, bce_ohem, criterion_margin_focal_binary_cross_entropy, ce_oheb, bi_tempered_loss, \
    resampleloss


def get_loss(cfg):
    return globals().get(cfg.loss.name)(**cfg.loss.param)

