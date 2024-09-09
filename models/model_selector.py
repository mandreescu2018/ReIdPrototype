from .vit_model import build_transformer, build_transformer_local

# from .mobilenetV2 import MobileNetV2

__factory = {
    'vit_transformer': build_transformer,
    # 'mobilenet': MobileNetV2,
}

def get_model(cfg):
    model = __factory[cfg.MODEL.NAME](cfg)
    return model