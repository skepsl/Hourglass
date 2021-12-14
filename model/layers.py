import torch
from torch import nn
import yaml
from pathlib import Path
from collections import OrderedDict
import inspect
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from thop import profile
import logging
import pytorch_model_summary

logger = logging.getLogger(__name__)


class Concat(nn.Module):
    def __init__(self, ch_in, dimension=1):
        super(Concat, self).__init__()
        self.dimension = dimension
        self._ch_out = np.sum(np.array(ch_in))

    @property
    def ch_out(self):
        return self._ch_out

    def forward(self, x):
        return torch.cat(x, self.dimension)


class Upsample(nn.Module):
    def __init__(self, ch_in, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(size=None, scale_factor=scale_factor, mode=mode)
        self._ch_out = ch_in

    @property
    def ch_out(self):
        return self._ch_out

    def forward(self, x):
        return self.upsample(x)


class Conv2D(nn.Module):
    def __init__(self, ch_in, ch_out, kernel=1, stride=1, bias=True, batch_norm=True, activation='leaky_relu',
                 activation_param=0.1):
        super(Conv2D, self).__init__()
        self._ch_out = ch_out
        self.conv = nn.Conv2d(ch_in, ch_out, (kernel, kernel), (stride, stride), padding=(kernel - 1) // 2, bias=bias)
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(ch_out)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(activation_param)
        else:
            self.activation = nn.Identity()

    @property
    def ch_out(self):
        return self._ch_out

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, ch_in, ch_out, shortcut=True, activation='leaky_relu', activation_param=0.1):
        super(Bottleneck, self).__init__()
        self._ch_out = ch_out
        ch_bott = int(ch_out / 2)
        self.shortcut = shortcut
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=activation_param)
        else:
            self.activation = nn.Identity()
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.conv1 = Conv2D(ch_in, ch_bott, kernel=1, stride=1, batch_norm=False, activation='')
        self.bn2 = nn.BatchNorm2d(ch_bott)
        self.conv2 = Conv2D(ch_bott, ch_bott, kernel=3, stride=1, batch_norm=False, activation='')
        self.bn3 = nn.BatchNorm2d(ch_bott)
        self.conv3 = Conv2D(ch_bott, ch_out, kernel=1, stride=1, batch_norm=False, activation='')
        if (ch_in == ch_out) or (shortcut is False):
            self.shortcut_layer = nn.Identity()
        else:
            self.shortcut_layer = Conv2D(ch_in, ch_out, kernel=1, stride=1, batch_norm=False, activation='')

    @property
    def ch_out(self):
        return self._ch_out

    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)
        if self.shortcut:
            out += self.shortcut_layer(x)
        return out


class Hourglass(nn.Module):
    def __init__(self, ch_in, inner_ch_list):
        super(Hourglass, self).__init__()
        self._ch_out = ch_in
        inner_ch = inner_ch_list[0]
        next_inner_ch_list = inner_ch_list[1:]
        self.shortcut = Bottleneck(ch_in, ch_in)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer_in = Bottleneck(ch_in, inner_ch)
        if next_inner_ch_list:
            self.inner_hourglass = Hourglass(inner_ch, next_inner_ch_list)
        else:
            self.inner_hourglass = Bottleneck(inner_ch, inner_ch)
        self.layer_out = Bottleneck(inner_ch, ch_in)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    @property
    def ch_out(self):
        return self._ch_out

    def forward(self, x):
        out = self.down(x)
        out = self.layer_in(out)
        out = self.inner_hourglass(out)
        out = self.layer_out(out)
        out = self.up(out)
        return out + self.shortcut(x)


class YAMLModel(nn.Module):
    def __init__(self, model_name):
        super(YAMLModel, self).__init__()
        self.module_list = nn.ModuleList()
        self.inputs = OrderedDict()
        self.layers = OrderedDict()
        self.outputs = OrderedDict()
        self.model_name = model_name
        file_dir = Path(__file__).parents[0].resolve() / 'yaml_model'
        self.yaml_file = file_dir / (model_name + '.yaml')
        with open(self.yaml_file) as f:
            self.yaml = yaml.safe_load(f)
        inputs, outputs, layers = self.yaml['inputs'], self.yaml['outputs'], self.yaml['layers']
        for id, args in inputs:
            self.inputs[id] = {'args': args}
        for id, parent, repeat, module, args in layers:
            ch_in = []
            parent = parent if isinstance(parent, list) else [parent]
            for idx, p in enumerate(parent):
                if isinstance(p, int):
                    parent[idx] = list(self.layers.items())[p][0]
                if parent[idx].startswith('input'):
                    ch_in.append(self.inputs[parent[idx]]['args'].get('ch_out', 0))
                else:
                    tmp = self.layers[parent[idx]]['module'][-1]
                    ch_in.append(tmp.ch_out if hasattr(tmp, 'ch_out') else 0)
            module_class = eval(module)
            sig = inspect.signature(module_class.__init__)
            ch_in_exist = False
            for param in sig.parameters.values():
                if param.name == 'ch_in':
                    ch_in_exist = True
                    args['ch_in'] = ch_in if len(ch_in) > 1 else ch_in[0]
            module_seq = []
            for r in range(repeat):
                module_seq.append(module_class(**args))
                if ch_in_exist and hasattr(module_seq[-1], 'ch_out'):
                    args['ch_in'] = module_seq[-1].ch_out
            module_instance = nn.Sequential(*module_seq)
            self.module_list.append(module_instance)
            self.layers[id] = {'parent': parent, 'module': module_instance}
        for id, parent in outputs:
            self.outputs[id] = {'parent': parent}

    def forward(self, x):
        x = (x,) if isinstance(x, torch.Tensor) else x
        for key, y in zip(self.inputs, x):
            self.inputs[key]['out'] = y
        for layer in self.layers.values():
            parent = layer['parent']
            module = layer['module']
            layer_in = []
            for p in parent:
                if p.startswith('input'):
                    layer_in.append(self.inputs[p]['out'])
                else:
                    layer_in.append(self.layers[p]['out'])
            layer_in = layer_in[0] if len(layer_in) == 1 else layer_in
            layer_out = module(layer_in)
            layer['out'] = layer_out
        out = []
        for id in self.outputs:
            out.append(self.layers[self.outputs[id]['parent']]['out'])
        out = tuple(out) if len(out) > 1 else out[0]
        return out

    def write_to_tensorboard(self, writer, input):
        writer.add_graph(self, input)

    def profile(self, input):
        macs, params = profile(model, inputs=(input,), verbose=False)
        num_inference = 100
        logger.info("Start profiling")
        start_time = time.time()
        for _ in tqdm(range(num_inference)):
            model(input)
        total_elapsed_time = time.time() - start_time
        inference_per_second = num_inference / total_elapsed_time
        logger.info(f"Input tensor shape: {tuple(input.shape)}")
        logger.info(f"Inference per second: {inference_per_second:.3f}Hz")
        logger.info(f"MACs: {(macs/1000000000):.3f}G")
        logger.info(f"Parameters: {params/1000000:.3f}M")

    def save(self, save_name):
        file_dir = Path(__file__).parents[0].resolve() / 'saved_model'
        torch.save(self.state_dict(), file_dir / (self.model_name + '_' + save_name))

    def load(self, save_name):
        file_dir = Path(__file__).parents[0].resolve() / 'saved_model'
        self.load_state_dict(torch.load(file_dir / (self.model_name + '_' + save_name)))


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    # create model
    # model = YAMLModel('yolov3').to(device)
    model = YAMLModel('hourglass')
    # export model summary to tensorboard
    # input = torch.rand((4, 24, 640, 640)).to(device)
    # sum_writer = SummaryWriter('./run')
    # model.write_to_tensorboard(sum_writer, input)
    # sum_writer.close()

    # model profiling
    # input = torch.rand((1, 24, 640, 640)).to(device)
    # pred=model(input)
    # model.profile(input)

    # save and load test
    # model.save('test')
    # model.load('test')
    print(pytorch_model_summary.summary(model, torch.zeros(1, 24, 640, 640), show_input=False))
























