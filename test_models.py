import argparse
import inspect
from config import cfg
# from models.models_configurations import model_cfg
from models.simple_model import SimpleReIDModel
from models.backbones.resnet_backbone import ResNet_Backbone
from models.vit_model import build_transformer
from datasets import make_dataloader
from models.model_selector import ModelLoader

def get_required_num_inputs(model):
    signature = inspect.signature(model.forward)
    params = signature.parameters
    required_params = [p for p in params.values() if p.default == inspect.Parameter.empty]
    return len(required_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    # parser.add_argument(
    #     "--config_file", default="configurations/Trans_ReID/Market/vit_base.yml", help="path to config file", type=str
    # )
    parser.add_argument(
        "--config_file", default="configurations/bag_of_tricks/Market/bag_of_tricks.yml", help="path to config file", type=str
    )
    # "configurations\Trans_ReID\Market\vit_base.yml"
    
    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader(cfg)
    
    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS = number_of_cameras
    cfg.DATASETS.NUMBER_OF_TRACKS = number_of_camera_tracks
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num

    cfg.MODEL.PRETRAIN_CHOICE = 'cross_domain'

    # Model
    model_loader = ModelLoader(cfg)
    model_loader.load_param()
    # print(model_loader.model)

    # model = build_transformer(cfg)
    # num_required_inputs = get_required_num_inputs(model)
    # print(f"The vit model requires {num_required_inputs} mandatory inputs.")



    # if args.config_file:
    #     model_cfg.merge_from_file(args.config_file)
    
    # print(len(model_cfg.MODEL.LAYERS))
    # print(model_cfg)

    # model = ModelBuilder()
    # for layer in model_cfg.MODEL.LAYERS:
    #     model.add_layer(layer["type"], **layer)
    
    # print(model.build())
        
    
    # layer_factory = LayerFactory.create_layer("linear", in_features=10, out_features=5)
    # print(layer_factory)