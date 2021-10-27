import math
import torch

def drop_connect(inputs, p, training):
    if not training:
        return inputs
    batch_size = inputs.size(0)
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, is_bn=True, is_act=True):
        super(ConvBlock, self).__init__()
        self._is_act = is_act
        self._is_bn = is_bn
        self._conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=1, bias=False)
        self._bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self._act = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        if self._is_bn:
            x = self._bn(x)
        if self._is_act:
            x = self._act(x)
        return x

class SEModule(torch.nn.Module):
    def __init__(self, in_channels, squeeze_channels, swish):
        super(SEModule, self).__init__()
        self._pool = torch.nn.AdaptiveAvgPool2d(1)
        self._se = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=squeeze_channels),
            swish,
            torch.nn.Linear(in_features=squeeze_channels, out_features=in_channels),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        x_pool = self._pool(x).view(b, c) # (b, c)
        return x * self._se(x_pool).view(b, c, 1, 1)


class MBConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=None, drop_connect_rate=None, batch_norm_momentum=0.99, batch_norm_epsilon=1e-3):
        super(MBConvBlock, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._drop_connect_rate = drop_connect_rate

        self._id_skip = (drop_connect_rate is not None) and (0 < drop_connect_rate <= 1)
        self._has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self._swish = torch.nn.SiLU()

        bn_mom = 1 - batch_norm_momentum
        bn_eps = batch_norm_epsilon
        mid_channels = int(in_channels * expand_ratio)

        # Expansion phase
        self._expand_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1, stride=1, padding=0, groups=1, is_bn=False, is_act=False
        )
        self._bn0 = torch.nn.BatchNorm2d(num_features=mid_channels, momentum=bn_mom, eps=bn_eps)
        # Depthwise convolution phase
        self._depthwise_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2), groups=mid_channels, is_bn=False, is_act=False
        )
        self._bn1 = torch.nn.BatchNorm2d(num_features=mid_channels, momentum=bn_mom, eps=bn_eps)
        # Squeeze and Excitation layer
        if self._has_se:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self._se = SEModule(in_channels=mid_channels, squeeze_channels=num_squeezed_channels, swish=self._swish)
        # Pointwise convolution phase
        self._project_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1, stride=1, padding=0, groups=1, is_bn=False, is_act=False
        )
        self._bn2 = torch.nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self._shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, inputs):
        x = inputs
        x = self._expand_conv(inputs)
        x = self._bn0(x)
        x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        if self._has_se:
            x = self._se(x)

        x = self._project_conv(x)
        x = self._bn2(x)

        if self._id_skip and self._stride == 1 and self._in_channels == self._out_channels:
            x = drop_connect(x, p=self._drop_connect_rate, training=self.training)

        x += self._shortcut(inputs)
        return x


class MBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate, num_layers):
        super(MBLayer, self).__init__()
        strides = [stride] + [1]*(num_layers-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate))
            in_channels = out_channels
        self._layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)

class EfficientNet(torch.nn.Module):
    def __init__(self, layer_params):
        super(EfficientNet, self).__init__()
        self._layers = torch.nn.ModuleList()
        self._layers.append(
            ConvBlock(
                in_channels=layer_params[0]['in_channels'],
                out_channels=layer_params[0]['out_channels'], 
                kernel_size=layer_params[0]['kernel_size'],
                stride=layer_params[0]['stride'],
                padding=layer_params[0]['padding'],
                groups=layer_params[0]['groups']
            )
        )
        for i in range(1, len(layer_params)):
            self._layers.append(
                MBLayer(
                    in_channels=layer_params[i]['in_channels'],
                    out_channels=layer_params[i]['out_channels'],
                    kernel_size=layer_params[i]['kernel_size'],
                    stride=layer_params[i]['stride'],
                    expand_ratio=layer_params[i]['expand_ratio'],
                    se_ratio=layer_params[i]['se_ratio'],
                    drop_connect_rate=layer_params[i]['drop_connect_rate'],
                    num_layers=layer_params[i]['num_layers']
                )
            )

    def forward(self, x):
        c_0 = self._layers[0](x)
        c_1 = self._layers[1](c_0)
        c_2 = self._layers[2](c_1)
        c_3 = self._layers[3](c_2)
        c_4 = self._layers[4](c_3)
        c_5 = self._layers[5](c_4)
        c_6 = self._layers[6](c_5)
        c_7 = self._layers[7](c_6)

        return c_7, c_6, c_5, c_4, c_3

def efficient_net(mode='b0'):
    coefficient_params = {
        'b0': (1.0, 1.0),
        'b1': (1.0, 1.1),
        'b2': (1.1, 1.2),
        'b3': (1.2, 1.4),
        'b4': (1.4, 1.8),
        'b5': (1.6, 2.2),
    }
    width_coefficient = coefficient_params[mode][0]
    depth_coefficient = coefficient_params[mode][1]
    layer_params = [
        {'in_channels': 3, 'out_channels': round_filters(32, width_coefficient), 'kernel_size': 3, 'stride': 2, 'padding': 1, 'groups':1},
        {'in_channels': round_filters(32, width_coefficient), 'out_channels': round_filters(16, width_coefficient), 'kernel_size': 3, 'stride': 1, 'expand_ratio': 1, 'se_ratio': None, 'drop_connect_rate': None, 'num_layers': round_repeats(1, depth_coefficient)},
        {'in_channels': round_filters(16, width_coefficient), 'out_channels': round_filters(24, width_coefficient), 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': None, 'drop_connect_rate': None, 'num_layers': round_repeats(2, depth_coefficient)},
        {'in_channels': round_filters(24, width_coefficient), 'out_channels': round_filters(40, width_coefficient), 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': None, 'drop_connect_rate': None, 'num_layers': round_repeats(2, depth_coefficient)},
        {'in_channels': round_filters(40, width_coefficient), 'out_channels': round_filters(80, width_coefficient), 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'num_layers': round_repeats(3, depth_coefficient)},
        {'in_channels': round_filters(80, width_coefficient), 'out_channels': round_filters(112, width_coefficient), 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'num_layers': round_repeats(3, depth_coefficient)},
        {'in_channels': round_filters(112, width_coefficient), 'out_channels': round_filters(192, width_coefficient), 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'num_layers': round_repeats(4, depth_coefficient)},
        {'in_channels': round_filters(192, width_coefficient), 'out_channels': round_filters(320, width_coefficient), 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'num_layers': round_repeats(1, depth_coefficient)}
    ]
    model = EfficientNet(layer_params)
    return model

def round_filters(filters, width_coefficient=None):
    if not width_coefficient:
        return filters
    divisor = 8
    filters *= width_coefficient
    new_filters = int(filters + divisor / 2) // divisor * divisor
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient=None):
    if not depth_coefficient:
        return repeats
    return int(math.ceil(depth_coefficient * repeats))
