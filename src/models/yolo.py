import torch
import torch.nn as nn
import yaml
import os
import logging
import math
import requests
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)


class ConvBNActivation(nn.Module):
    """
    Convolutional layer with batch normalization and activation function
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, bias=False, activation=nn.LeakyReLU(0.1, inplace=True)):
        super(ConvBNActivation, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Bottleneck(nn.Module):
    """
    CSP Bottleneck with 3 convolutions
    """

    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNActivation(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNActivation(hidden_channels, out_channels, 3, 1, 1, groups=groups)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))


class CSPLayer(nn.Module):
    """
    CSP (Cross Stage Partial) layer with 3 convolutions
    """

    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcut=True, groups=1, expansion=0.5):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNActivation(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNActivation(in_channels, hidden_channels, 1, 1)
        self.conv3 = ConvBNActivation(2 * hidden_channels, out_channels, 1, 1)

        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, groups, 1.0)
            for _ in range(num_bottlenecks)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.m(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling layer used in YOLOv3-SPP
    """

    def __init__(self, in_channels, out_channels, kernels=(5, 9, 13)):
        super(SPP, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNActivation(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNActivation(hidden_channels * (len(kernels) + 1), out_channels, 1, 1)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernels
        ])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    """
    Focus width and height information into channel space
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super(Focus, self).__init__()
        self.conv = ConvBNActivation(
            in_channels * 4, out_channels, kernel_size, stride,
            padding, groups=groups
        )

    def forward(self, x):
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right),
            dim=1
        )
        return self.conv(x)


class Backbone(nn.Module):
    """
    YOLOv5 backbone (CSPDarknet)
    """

    def __init__(self, input_channels=3, depth_multiple=1.0, width_multiple=1.0):
        super(Backbone, self).__init__()
        # Scaling factors
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

        # Base channels
        base_channels = int(64 * width_multiple)
        num_repeats = max(round(3 * depth_multiple), 1)

        # Build stages
        self.stem = Focus(input_channels, base_channels, 3)  # P1/2

        self.stage2 = nn.Sequential(
            ConvBNActivation(base_channels, base_channels * 2, 3, 2, 1),  # P2/4
            CSPLayer(base_channels * 2, base_channels * 2, num_repeats)
        )

        self.stage3 = nn.Sequential(
            ConvBNActivation(base_channels * 2, base_channels * 4, 3, 2, 1),  # P3/8
            CSPLayer(base_channels * 4, base_channels * 4, num_repeats * 3)
        )

        self.stage4 = nn.Sequential(
            ConvBNActivation(base_channels * 4, base_channels * 8, 3, 2, 1),  # P4/16
            CSPLayer(base_channels * 8, base_channels * 8, num_repeats * 3)
        )

        self.stage5 = nn.Sequential(
            ConvBNActivation(base_channels * 8, base_channels * 16, 3, 2, 1),  # P5/32
            SPP(base_channels * 16, base_channels * 16),
            CSPLayer(base_channels * 16, base_channels * 16, num_repeats)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x_p3 = self.stage3(x)
        x_p4 = self.stage4(x_p3)
        x_p5 = self.stage5(x_p4)
        return (x_p3, x_p4, x_p5)


class Neck(nn.Module):
    """
    YOLOv5 neck (PANet)
    """

    def __init__(self, width_multiple=1.0):
        super(Neck, self).__init__()

        base_channels = int(64 * width_multiple)

        # Top-down pathway
        self.lateral_conv5 = ConvBNActivation(base_channels * 16, base_channels * 8, 1, 1)
        self.fpn_conv5 = CSPLayer(base_channels * 16, base_channels * 8, 1, False)

        self.lateral_conv4 = ConvBNActivation(base_channels * 8, base_channels * 4, 1, 1)
        self.fpn_conv4 = CSPLayer(base_channels * 8, base_channels * 4, 1, False)

        # Bottom-up pathway
        self.downsample_conv4 = ConvBNActivation(base_channels * 4, base_channels * 4, 3, 2, 1)
        self.pan_conv4 = CSPLayer(base_channels * 8, base_channels * 8, 1, False)

        self.downsample_conv5 = ConvBNActivation(base_channels * 8, base_channels * 8, 3, 2, 1)
        self.pan_conv5 = CSPLayer(base_channels * 16, base_channels * 16, 1, False)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):
        p3, p4, p5 = inputs

        # Top-down pathway (feature pyramid)
        p5_td = self.lateral_conv5(p5)
        p5_up = self.upsample(p5_td)
        p4 = torch.cat([p4, p5_up], dim=1)
        p4 = self.fpn_conv5(p4)

        p4_td = self.lateral_conv4(p4)
        p4_up = self.upsample(p4_td)
        p3 = torch.cat([p3, p4_up], dim=1)
        p3 = self.fpn_conv4(p3)

        # Bottom-up pathway (additional feature levels)
        p3_down = self.downsample_conv4(p3)
        p4 = torch.cat([p3_down, p4_td], dim=1)
        p4 = self.pan_conv4(p4)

        p4_down = self.downsample_conv5(p4)
        p5 = torch.cat([p4_down, p5_td], dim=1)
        p5 = self.pan_conv5(p5)

        return (p3, p4, p5)


class YOLOHead(nn.Module):
    """
    YOLOv5 detection head
    """

    def __init__(self, width_multiple=1.0, num_classes=80, num_anchors=3):
        super(YOLOHead, self).__init__()

        base_channels = int(64 * width_multiple)

        # Output channels: [x, y, w, h, objectness, num_classes] for each anchor
        output_channels = num_anchors * (5 + num_classes)

        # Detection heads
        self.detection_head_p3 = nn.Conv2d(base_channels * 4, output_channels, 1)
        self.detection_head_p4 = nn.Conv2d(base_channels * 8, output_channels, 1)
        self.detection_head_p5 = nn.Conv2d(base_channels * 16, output_channels, 1)

    def forward(self, inputs):
        p3, p4, p5 = inputs

        # Apply detection heads
        output_p3 = self.detection_head_p3(p3)
        output_p4 = self.detection_head_p4(p4)
        output_p5 = self.detection_head_p5(p5)

        return (output_p3, output_p4, output_p5)


class YOLOv5(nn.Module):
    """
    YOLOv5 object detection model
    """

    def __init__(self, config=None, num_classes=80, input_channels=3):
        super(YOLOv5, self).__init__()

        # Default configuration for YOLOv5s
        if config is None:
            config = {
                'width_multiple': 0.5,  # YOLOv5s width multiple
                'depth_multiple': 0.33,  # YOLOv5s depth multiple
                'input_channels': input_channels,
                'num_classes': num_classes
            }

        width_multiple = config.get('width_multiple', 0.5)
        depth_multiple = config.get('depth_multiple', 0.33)
        input_channels = config.get('input_channels', 3)
        num_classes = config.get('num_classes', 80)

        # Build backbone, neck, and head
        self.backbone = Backbone(input_channels, depth_multiple, width_multiple)
        self.neck = Neck(width_multiple)
        self.head = YOLOHead(width_multiple, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Backbone
        features = self.backbone(x)

        # Neck
        features = self.neck(features)

        # Head
        outputs = self.head(features)

        # Process the outputs for detection
        return self._process_outputs(outputs, x.shape)

    def _process_outputs(self, outputs, input_shape):
        """
        Process the raw outputs into a format suitable for inference

        Args:
            outputs: Raw outputs from the model
            input_shape: Shape of the input tensor

        Returns:
            torch.Tensor: Processed predictions
        """
        batch_size = input_shape[0]

        # Process each feature map
        results = []
        for output in outputs:
            batch, channels, height, width = output.shape

            # Reshape output to [batch, num_anchors, height, width, num_classes + 5]
            # where 5 is [x, y, w, h, objectness]
            output = output.view(batch, 3, -1, height, width)
            output = output.permute(0, 1, 3, 4, 2).contiguous()
            output = output.view(batch, -1, output.shape[-1])

            results.append(output)

        # Concatenate predictions from all feature maps
        predictions = torch.cat(results, dim=1)

        return predictions


def create_model(model_config, num_classes=80, input_channels=3, pretrained=True, pretrained_weights=None):
    """
    Create YOLOv5 model

    Args:
        model_config (dict): Model configuration
        num_classes (int): Number of classes
        input_channels (int): Number of input channels
        pretrained (bool): Whether to use pretrained weights
        pretrained_weights (str): Path to pretrained weights

    Returns:
        YOLOv5: Model
    """
    # Extract model size
    model_size = model_config['model']['version']

    # Define width and depth multipliers based on model size
    size_to_multipliers = {
        'n': {'width_multiple': 0.25, 'depth_multiple': 0.33},  # YOLOv5n (nano)
        's': {'width_multiple': 0.5, 'depth_multiple': 0.33},  # YOLOv5s (small)
        'm': {'width_multiple': 0.75, 'depth_multiple': 0.67},  # YOLOv5m (medium)
        'l': {'width_multiple': 1.0, 'depth_multiple': 1.0},  # YOLOv5l (large)
        'x': {'width_multiple': 1.25, 'depth_multiple': 1.33}  # YOLOv5x (xlarge)
    }

    # Get multipliers for the specified model size
    multipliers = size_to_multipliers.get(model_size, size_to_multipliers['s'])

    # Create model configuration
    config = {
        'width_multiple': multipliers['width_multiple'],
        'depth_multiple': multipliers['depth_multiple'],
        'input_channels': input_channels,
        'num_classes': num_classes
    }

    # Create model
    model = YOLOv5(config=config)

    # Load pretrained weights if specified
    if pretrained:
        if pretrained_weights is not None:
            # Check if file exists
            weights_path = Path(pretrained_weights)
            if weights_path.exists():
                logger.info(f"Loading pretrained weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location='cpu')

                # Check if it's a checkpoint or just weights
                if isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']

                try:
                    # Try to load weights
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully loaded pretrained weights")
                except Exception as e:
                    logger.warning(f"Failed to load pretrained weights: {e}")
            else:
                logger.warning(f"Pretrained weights file not found: {weights_path}")
                # Download pretrained weights if file doesn't exist
                _download_pretrained_weights(model_size, weights_path)

                if weights_path.exists():
                    logger.info(f"Loading downloaded pretrained weights from {weights_path}")
                    model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)

    return model


def _download_pretrained_weights(model_size, save_path):
    """
    Download pretrained weights for YOLOv5 model

    Args:
        model_size (str): Model size (n, s, m, l, x)
        save_path (Path): Path to save weights
    """
    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Download weights
    url = f"https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5{model_size}.pt"
    logger.info(f"Downloading pretrained weights from {url}")

    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(save_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            logger.error("Error downloading weights")
            save_path.unlink()  # Remove partially downloaded file
        else:
            logger.info(f"Successfully downloaded pretrained weights to {save_path}")

    except Exception as e:
        logger.error(f"Failed to download pretrained weights: {e}")
        if save_path.exists():
            save_path.unlink()  # Remove partially downloaded file


if __name__ == "__main__":
    # Test model creation
    model_config = {
        'model': {
            'version': 's',
            'num_classes': 20
        }
    }

    model = create_model(model_config, num_classes=20, pretrained=False)
    print(model)

    # Test forward pass
    x = torch.randn(1, 3, 416, 416)
    output = model(x)
    print(f"Output shape: {output.shape}")