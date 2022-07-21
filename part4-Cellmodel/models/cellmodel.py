import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class CELLResNet50(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(pooled_features))
        return output

    @property
    def net(self):
        return self.model


class CELLXception(nn.Module):
    def __init__(self, model_name='xception', out_dim=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(pooled_features))
        return output

    @property
    def net(self):
        return self.model


class tfResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.addmodel = nn.Sequential(nn.BatchNorm1d(n_features), nn.Dropout(dropout), nn.ReLU(),
                                       nn.Linear(n_features, 512), nn.BatchNorm1d(512),
                                       nn.Dropout(dropout/2.0), nn.ReLU(), nn.Linear(512, out_dim))

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        # output = self.fc(self.dropout(pooled_features))
        output = self.addmodel(pooled_features)
        return output

    @property
    def net(self):
        return self.model