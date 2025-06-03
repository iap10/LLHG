import torch
import torchvision.models as models
from hopenet import Hopenet
import numpy as np
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import os
import time

# torch to onnx
def export_hopenet_onnx(pth_path, onnx_path):
    # load pretrained model
    resnet50 = models.resnet50(pretrained=False)
    model = Hopenet(resnet50, 66)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'), strict=False)
    model.eval()

    # dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'],
                      output_names=['yaw', 'pitch', 'roll'],
                      dynamic_axes={'input': {0: 'batch_size'}},
                      opset_version=11)

# onnx to trt
def build_trt_engine(onnx_path, engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 28  # 256MB

    # define network
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parsing .onnx
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX file.")

    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # save as .engine
    engine = builder.build_engine(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, default='hopenet_robust_alpha1.pth')
    parser.add_argument('--onnx', type=str, default='hopenet.onnx')
    parser.add_argument('--engine', type=str, default='hopenet.engine')
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        export_hopenet_onnx(args.pth, args.onnx)

    if not os.path.exists(args.engine):
        build_trt_engine(args.onnx, args.engine)
