import yaml
import json
import os
import glob


class Element:
    def __repr__(self):
        return ', '.join(['{}: {}'.format(k, v) for k, v in self.__dict__.items()])


class Basic(Element):
    def __init__(self, dict):
        self.seed = int(dict.get('seed', '233'))
        self.GPU = str(dict.get('GPU', '0'))
        self.debug = dict.get('debug', False)


class Data(Element):
    def __init__(self, dict):
        self.root_data = dict.get('root_data', 'none')
        self.root_mask = dict.get('root_mask', 'none')
        self.save_dir = dict.get('save_dir', 'none')



class MILModel(Element):
    def __init__(self, dict):
        self.name1 = dict.get('name1', 'resnet50')
        self.path1 = dict.get('pth_path1', '')
        self.name2 = dict.get('name2', 'resnet50')
        self.path2 = dict.get('pth_path2', '')
        self.frac = dict.get('frac', 1)


class CellModel(Element):
    def __init__(self, dict):
        self.name = dict.get('name', 'Cell_resnet50d')
        self.path = dict.get('pth_path', '')
        self.frac = dict.get('frac', 1)


class Model(Element):
    def __init__(self, dict):
        self.name = dict.get('name', '')
        self.param = dict.get('param', {'pretrained': True})


class Config:
    def __init__(self, dict):
        self.param = dict
        self.basic = Basic(dict.get('basic', {}))
        self.data = Data(dict.get('data', {}))
        self.MILmodel = MILModel(dict.get('MILmodel', {}))
        self.CellModel = CellModel(dict.get('CellModel', {}))
        self.model = Model(dict.get('Model', {}))

    def __repr__(self):
        return '\t\n'.join(['{}: {}'.format(k, v) for k, v in self.__dict__.items()])

    def dump_json(self, file_path):
        with open(file_path, 'w') as fp:
            json.dump(self.param, fp, indent=4)

    def to_flatten_dict(self):
        ft = {}
        for k, v in self.param.items():
            for kk, vv in v.items():
                if type(vv) in [dict, list]:
                    vv = str(vv)
                ft[f'{k}.{kk}'] = vv
        return ft

    @staticmethod
    def load_json(file_path):
        with open(file_path) as fp:
            data = json.load(fp)
        return Config(data)

    @staticmethod
    def load(file_path):
        with open(file_path) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
        return Config(data)


def get_config(name):
    return Config.load(os.path.dirname(os.path.realpath(__file__)) + '/' + name)


if __name__ == '__main__':
    args = get_config('example.yaml')

