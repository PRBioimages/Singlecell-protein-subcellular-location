from models.resnetd import MILResNet50D
from models.xception import MILXception
from models.attention_model import AttentionResNet50D, Custom_ATTResNet50D, GAttentionResNet50D
from models.SA_model import SA_attentionResNet, SA_MaxResNet, SA_Max_CalibrationResNet
from models.Transformer import Transformer_ResNet50D
from models.imp_Transformer import imp_Transformer_ResNet50D
# from models.CLAM import CLAM_SB, CLAM_MB
from models.cellmodel import *
from configs import Config


def get_model(cfg: Config, pretrained='imagenet'):
    if cfg.model.name in ['resnet200d', 'resnet152d', 'resnet101d', 'resnet50d', 'resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a MILResNet, mdl_name: {cfg.model.name}, pool: {pool}')
        return MILResNet50D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['xception']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a MILXception, mdl_name: {cfg.model.name}, pool: {pool}')
        return MILXception(pretrained=cfg.model.param['pretrained'], model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['attention_xception','attention_resnet200d', 'attention_resnet152d', 'attention_resnet101d', 'attention_resnet50d', 'attention_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a Attention-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return AttentionResNet50D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['custonattention_xception','custonattention_resnet200d', 'custonattention_resnet152d', 'custonattention_resnet101d', 'custonattention_resnet50d', 'custonattention_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a Custom_ATTResNet50D, mdl_name: {cfg.model.name}, pool: {pool}')
        return Custom_ATTResNet50D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['SA_attention_xception','SA_attention_resnet200d', 'SA_attention_resnet152d', 'SA_attention_resnet101d', 'SA_attention_resnet50d', 'SA_attention_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a SA_Attention-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return SA_attentionResNet(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['SA_Max_xception','SA_Max_resnet200d', 'SA_Max_resnet152d', 'SA_Max_resnet101d', 'SA_Max_resnet50d', 'SA_Max_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a SA_Max-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return SA_MaxResNet(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['SA_Max_Calibration_xception','SA_Max_Calibration_resnet200d', 'SA_Max_Calibration_resnet152d', 'SA_Max_Calibration_resnet101d', 'SA_Max_Calibration_resnet50d', 'SA_Max_Calibration_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a SA_Max_CalibrationResNet, mdl_name: {cfg.model.name}, pool: {pool}')
        return SA_Max_CalibrationResNet(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    elif cfg.model.name in ['Cell_resnet50d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a Cell-ResNet, mdl_name: {cfg.model.name}, pool: {pool}')
        return CELLResNet50(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0), out_dim=cfg.model.out_feature)
    elif cfg.model.name in ['Cell_xception']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a Cell-xception, mdl_name: {cfg.model.name}, pool: {pool}')
        return CELLXception(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0), out_dim=cfg.model.out_feature)
    elif cfg.model.name in ['gattention_resnet50d', 'gattention_xception']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a GAttention-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return GAttentionResNet50D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    elif cfg.model.name in ['transformer_xception','transformer_resnet200d', 'transformer_resnet152d', 'transformer_resnet101d', 'transformer_resnet50d', 'transformer_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a transformer-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return Transformer_ResNet50D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    elif cfg.model.name in ['imp_transformer_xception','imp_transformer_resnet200d', 'imp_transformer_resnet152d', 'imp_transformer_resnet101d', 'imp_transformer_resnet50d', 'imp_transformer_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a imp_transformer-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return imp_Transformer_ResNet50D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))

