import torch
import config as cfg
from efficient_net import efficient_net
from utils import Upsample, multi_apply, ResidualBlock

class FPN(torch.nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self._backbone = efficient_net(mode='b0')
        self._channels = [320, 192, 112, 80, 40]
        fpn_channels = 192

        self._in_layers = torch.nn.ModuleList()
        for feat_channels in self._channels:
            self._in_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=feat_channels, out_channels=fpn_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    torch.nn.BatchNorm2d(num_features=fpn_channels)
                )
            )

        self._upsampling = Upsample()
        self._relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        features = self._backbone(x)
        in_feats = []
        for feat, in_layer in zip(features, self._in_layers):
            in_feats.append(in_layer(feat))

        x = in_feats[0]
        p_feats = [x]
        for i in range(len(in_feats)-1):
            x = self._relu(self._upsampling(x, fsize=features[i+1].size()[2:]) + in_feats[i+1])
            p_feats.append(x)

        return p_feats[::-1]

class Model(torch.nn.Module):
    def __init__(self, n_classes=len(cfg.categories)):
        super(Model, self).__init__()
        in_channels = 192
        out_channels = 128
        n_p = 5
        self._fpn = FPN()
        self._n_classes = n_classes
        self._n_reg = cfg.reg_max + 1

        self._classifiers = torch.nn.ModuleList()
        self._regressors = torch.nn.ModuleList()
        for _ in range(n_p):
            self._classifiers.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.Conv2d(in_channels=out_channels, out_channels=self._n_classes, kernel_size=1, stride=1, padding=0, bias=False),
            ))
            self._regressors.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.Conv2d(in_channels=out_channels, out_channels=4 * self._n_reg, kernel_size=1, stride=1, padding=0, bias=False),
            ))

    def _cls_single_process(self, classifier, feature):
        return classifier(feature)

    def _reg_single_process(self, regressor, feature):
        return regressor(feature)

    def forward(self, x):
        fpn = self._fpn(x)
        cls_head_outs = multi_apply(self._cls_single_process, self._classifiers, fpn)
        reg_head_outs = multi_apply(self._reg_single_process, self._regressors, fpn)

        batch = fpn[0].size(0)
        cls_out = torch.cat([cls_head.reshape(batch, self._n_classes, -1) for cls_head in cls_head_outs], dim=-1)
        reg_out = torch.cat([reg_head.reshape(batch, 4, self._n_reg, -1) for reg_head in reg_head_outs], dim=-1)

        return cls_out, reg_out
        



# if __name__ == '__main__':
#     import onnx
#     import onnxruntime
#     import time, os

#     use_gpu = torch.cuda.is_available()

#     x = torch.randn(1, 3, cfg.size[1], cfg.size[0]).cuda()
#     model = torch.nn.DataParallel(Model()).cuda()
#     model = model.eval()

#     torch.onnx.export(
#         model.module, x,
#         './model/model.onnx',
#         export_params=True,
#         opset_version=12,
#         do_constant_folding=True,
#         input_names = ['input'],
#         output_names = ['output'],
#         dynamic_axes={
#             'input' : {0 : 'batch_size', 2: 'height', 3: 'width'}
#         }
#     )

#     # inference
#     onnx_model = onnx.load('./model/model.onnx')
#     onnx.checker.check_model(onnx_model)
#     ort_session = onnxruntime.InferenceSession('./model/model.onnx')

#     ort_inputs = {ort_session.get_inputs()[0].name: x.detach().cpu().numpy()}
#     ort_outputs = ort_session.run(None, ort_inputs)

#     for o in ort_outputs:
#         print(o.shape)