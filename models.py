"""
Neural network architectures for gait analysis.

CRITICAL: These classes must EXACTLY match the code used during training,
or pretrained weights will silently fail to load (keys won't match).

Architecture for ElderNet inference:
  feature_extractor (ResNet backbone)
       ↓  [1024-dim flat vector]
  fc (LinearLayers: 1024 → 512 → 256 → 128)
       ↓  [128-dim representation]
  classifier (Classifier: 128 → 2)        ← for gait detection
  regressor  (Regressor: 128 → 1 via mu)  ← for speed/cadence/etc.

Weight key mapping:
  feature_extractor.layer1.0.weight  ...  (backbone)
  fc.linear1.weight                       (1024 → 512)
  fc.linear2.weight                       (512 → 256)
  fc.linear3.weight                       (256 → 128)
  classifier.linear1.weight               (128 → 2)
  regressor.linear_layers.0.weight        (128 → 64, if num_layers=1)
  regressor.mu.weight                     (64 → 1)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Head modules (must match training code exactly)
# ============================================================================

class Classifier(nn.Module):
    """Single linear layer for classification."""
    def __init__(self, input_size=1024, output_size=2):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear1(x)


class EvaClassifier(nn.Module):
    """Two-layer classifier with ReLU."""
    def __init__(self, input_size=1024, nn_size=512, output_size=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Regressor(nn.Module):
    """
    Configurable regression head with optional uncertainty.

    Weight keys (example with num_layers=1, input=128):
        regressor.linear_layers.0.weight  (128 → 64)
        regressor.linear_layers.0.bias
        regressor.bn_layers.0.weight      (if batch_norm=True)
        regressor.bn_layers.0.bias
        regressor.mu.weight               (64 → 1)
        regressor.mu.bias
        regressor.var.weight              (64 → 1)
        regressor.var.bias
    """
    def __init__(self, input_size=1024, output_size=1, uncertainty=False,
                 max_mu=2.0, max_var=50.0, num_layers=1, batch_norm=False):
        super(Regressor, self).__init__()
        self.uncertainty = uncertainty
        self.max_mu = max_mu
        self.max_var = max_var
        self.bn = batch_norm

        self.linear_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        hidden_size = input_size
        for i in range(num_layers):
            next_hidden_size = max(hidden_size // 2, 32)
            self.linear_layers.append(nn.Linear(hidden_size, next_hidden_size))
            if self.bn:
                self.bn_layers.append(nn.BatchNorm1d(next_hidden_size))
            hidden_size = next_hidden_size

        self.mu = nn.Linear(hidden_size, output_size)
        self.var = nn.Linear(hidden_size, output_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.bn:
            for linear, bn in zip(self.linear_layers, self.bn_layers):
                x = F.relu(bn(linear(x)))
        else:
            for linear in self.linear_layers:
                x = F.relu(linear(x))

        mu = self.max_mu * torch.sigmoid(self.mu(x))

        if self.uncertainty:
            var = self.max_var * torch.sigmoid(self.var(x)) + 1e-6
            return mu, var
        return mu


class LinearLayers(nn.Module):
    """
    ElderNet's FC adapter: 3 fixed linear layers halving dimensions.

    1024 → 512 → 256 → output_size (default 128)

    Weight keys:
        fc.linear1.weight  (1024 → 512)
        fc.linear2.weight  (512 → 256)
        fc.linear3.weight  (256 → 128)
    """
    def __init__(self, input_size=1024, output_size=128, non_linearity=False):
        super(LinearLayers, self).__init__()
        assert input_size / 4 > 0, "input size too small"
        assert output_size <= (input_size / 4), "output size needs to be smaller than input size/4"
        self.linear1 = torch.nn.Linear(input_size, int(input_size / 2))
        self.linear2 = torch.nn.Linear(int(input_size / 2), int(input_size / 4))
        self.linear3 = torch.nn.Linear(int(input_size / 4), output_size)
        self.relu = nn.ReLU()
        self.non_linearity = non_linearity
        weight_init(self)

    def forward(self, x):
        if self.non_linearity:
            fc1 = self.linear1(x)
            fc2 = self.linear2(self.relu(fc1))
            out = self.linear3(self.relu(fc2))
        else:
            fc1 = self.linear1(x)
            fc2 = self.linear2(fc1)
            out = self.linear3(fc2)
        return out


# ============================================================================
# Backbone
# ============================================================================

class Downsample(nn.Module):
    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1
        self.stride = factor
        self.channels = channels
        self.order = order
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0
        self.padding = int(order * (factor - 1) / 2)
        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer("kernel", kernel[None, None, :].repeat((channels, 1, 1)))

    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride,
                        padding=self.padding, groups=x.shape[1])


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                               padding, bias=False, padding_mode="circular")
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride,
                               padding, bias=False, padding_mode="circular")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)
        return x + identity


class Resnet(nn.Module):
    def __init__(self, n_channels=3, resnet_version=1, output_size=1, epoch_len=10,
                 is_mtl=False, is_simclr=False, is_classification=False,
                 is_regression=False, is_uncertain=False,
                 max_mu=None, max_var=None,
                 num_layers_regressor=None, batch_norm=False,
                 feature_extractor=nn.Sequential()):
        super(Resnet, self).__init__()
        self.output_size = output_size
        self.is_mtl = is_mtl
        self.is_simclr = is_simclr
        self.is_classification = is_classification
        self.is_regression = is_regression
        self.is_uncertain = is_uncertain
        self.max_mu = max_mu
        self.max_var = max_var
        self.num_layers_regressor = num_layers_regressor
        self.batch_norm = batch_norm

        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [(64,5,2,5,2,2),(128,5,2,5,2,2),(256,5,2,5,3,1),
                       (256,5,2,5,3,1),(512,5,0,5,3,1)]
            elif epoch_len == 10:
                cgf = [(64,5,2,5,2,2),(128,5,2,5,2,2),(256,5,2,5,5,1),
                       (512,5,2,5,5,1),(1024,5,0,5,3,1)]
            else:
                cgf = [(64,5,2,5,3,1),(128,5,2,5,3,1),(256,5,2,5,5,1),
                       (512,5,2,5,5,1),(1024,5,0,5,4,0)]
        else:
            cgf = [(64,5,2,5,3,1),(64,5,2,5,3,1),(128,5,2,5,5,1),
                   (128,5,2,5,5,1),(256,5,2,5,4,0)]

        in_channels = n_channels
        self.feature_extractor = feature_extractor
        for i, (out_ch, conv_k, n_res, res_k, down_f, down_o) in enumerate(cgf):
            self.feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(in_channels, out_ch, conv_k, n_res, res_k, down_f, down_o)
            )
            in_channels = out_ch

        if self.is_mtl:
            self.aot_h = Classifier(out_ch, output_size)
            self.scale_h = Classifier(out_ch, output_size)
            self.permute_h = Classifier(out_ch, output_size)
            self.time_w_h = Classifier(out_ch, output_size)
        elif self.is_simclr:
            self.classifier = Classifier(out_ch, output_size)
        elif self.is_classification:
            self.classifier = EvaClassifier(out_ch, output_size=output_size)
        elif self.is_regression:
            self.regressor = Regressor(
                input_size=out_ch, output_size=output_size,
                uncertainty=self.is_uncertain,
                max_mu=self.max_mu, max_var=self.max_var,
                num_layers=self.num_layers_regressor, batch_norm=self.batch_norm
            )

        weight_init(self)

    @staticmethod
    def make_layer(in_ch, out_ch, conv_k, n_res, res_k, down_f, down_o=1):
        pad_c = (conv_k - 1) // 2
        pad_r = (res_k - 1) // 2
        modules = [nn.Conv1d(in_ch, out_ch, conv_k, 1, pad_c, bias=False,
                             padding_mode="circular")]
        for _ in range(n_res):
            modules.append(ResBlock(out_ch, out_ch, res_k, 1, pad_r))
        modules += [nn.BatchNorm1d(out_ch), nn.ReLU(True),
                    Downsample(out_ch, down_f, down_o)]
        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)
        flat = feats.view(x.shape[0], -1)
        if self.is_mtl:
            return (self.aot_h(flat), self.scale_h(flat),
                    self.permute_h(flat), self.time_w_h(flat))
        elif self.is_simclr:
            return self.classifier(flat)
        elif self.is_classification:
            return self.classifier(flat)
        elif self.is_regression:
            return self.regressor(flat)


# ============================================================================
# ElderNet
# ============================================================================

class ElderNet(nn.Module):
    """
    ElderNet = ResNet feature extractor + LinearLayers adapter + task head.

    Data flow:
        input (B, 3, 300)
          → feature_extractor → (B, 1024, 1)
          → flatten → (B, 1024)
          → fc (LinearLayers) → (B, 128)     ← "representation"
          → classifier or regressor → output

    For gait detection (classification):
        pretrained weights have: fc.linear1/2/3 + classifier.linear1

    For speed/cadence/etc (regression):
        pretrained weights have: fc.linear1/2/3 + regressor.linear_layers.0 + regressor.mu
    """
    def __init__(self, feature_extractor, head='fc', non_linearity=True,
                 linear_model_input_size=1024, linear_model_output_size=50,
                 output_size=1,
                 is_mtl=False, is_simclr=False, is_classification=False,
                 is_dense=False, is_regression=False, is_uncertain=False,
                 max_mu=None, max_var=None,
                 num_layers_regressor=None, batch_norm=False):
        super(ElderNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.output_size = output_size
        self.is_mtl = is_mtl
        self.is_simclr = is_simclr
        self.is_classification = is_classification
        self.is_dense = is_dense
        self.is_regression = is_regression
        self.is_uncertain = is_uncertain
        self.head = head

        if self.is_regression:
            self.max_mu = max_mu
            self.max_var = max_var
            self.num_layers_regressor = num_layers_regressor
            self.batch_norm = batch_norm

        # Freeze backbone during SSL
        if self.is_mtl or self.is_simclr:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # FC adapter: 1024 → 512 → 256 → output_size (128)
        if self.head == 'fc':
            self.fc = LinearLayers(linear_model_input_size,
                                   linear_model_output_size, non_linearity)

        if self.is_mtl:
            self.aot_h = Classifier(linear_model_output_size, 2)
            self.scale_h = Classifier(linear_model_output_size, 2)
            self.permute_h = Classifier(linear_model_output_size, 2)
            self.time_w_h = Classifier(linear_model_output_size, 2)

        if self.is_classification:
            self.classifier = Classifier(linear_model_output_size, self.output_size)

        if self.is_regression:
            self.regressor = Regressor(
                input_size=linear_model_output_size,
                output_size=self.output_size,
                uncertainty=self.is_uncertain,
                max_mu=self.max_mu, max_var=self.max_var,
                num_layers=self.num_layers_regressor,
                batch_norm=self.batch_norm
            )

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.head == 'fc':
            representation = self.fc(features.view(x.shape[0], -1))
        elif self.head == 'unet':
            raise NotImplementedError("Unet head not used in inference pipeline")

        if self.is_mtl:
            return (self.aot_h(representation), self.scale_h(representation),
                    self.permute_h(representation), self.time_w_h(representation))
        elif self.is_simclr:
            return representation
        elif self.is_classification or self.is_dense:
            return self.classifier(representation)
        elif self.is_regression:
            return self.regressor(representation)


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(self, mode="fan_out", nonlinearity="relu"):
    set_seed()
    for m in self.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
