from .processor_prototype import ProcessorPrototype

__factory = {
    'vit_transformer': ProcessorPrototype,
    'vit_transformer_jpm': ProcessorPrototype,
    'mobilenet_v2': ProcessorPrototype,
    'resnet50': ProcessorPrototype,
    'simple_resnet50': ProcessorPrototype,
    'hacnn': ProcessorPrototype
}

def get_processor(cfg):
    proc = __factory[cfg.MODEL.NAME]
    return proc