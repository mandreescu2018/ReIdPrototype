import time
import torch
import argparse
from config import cfg
from utils import set_seeds
from datasets import make_dataloader
# from datasets.make_dataloader_trans import make_dataloader
from models import ModelLoader
from processors.processor_standard import ProcessorStandard
import onnxruntime as ort


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="configurations/Trans_ReID/Market/vit_base.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    set_seeds(cfg.SOLVER.SEED)

    # datasets related
    train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader(cfg)

    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS = number_of_cameras
    cfg.DATASETS.NUMBER_OF_TRACKS = number_of_camera_tracks
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num

    
    # Model
    cfg.MODEL.PRETRAIN_CHOICE = 'test'
    model_loader = ModelLoader(cfg)
    model_loader.load_param()
    
    # Dummy input
dummy_input = torch.randn(1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])
model_loader.model.to('cpu')  # Move model to CPU for ONNX export
# Export to ONNX
torch.onnx.export(
    model_loader.model,
    dummy_input,
    "vit_base_reid.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("✅ Exported to vit_reid.onnx")
    
    # proc = ProcessorStandard(cfg, 
    #                             model_loader.model, 
    #                             train_loader, 
    #                             test_loader)
    # start = time.perf_counter()
    
    # proc.inference()

    # print(f"Time taken: {time.perf_counter() - start}")


