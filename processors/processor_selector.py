from .processor_standard import ProcessorStandard

__factory = {
    'vit_transformer': ProcessorStandard,
    'vit_transformer_jpm': ProcessorStandard,
    'mobilenet_v2': ProcessorStandard,
    'resnet50': ProcessorStandard,
    'simple_resnet50': ProcessorStandard,
    'hacnn': ProcessorStandard
}

def get_processor(cfg):
    proc = __factory[cfg.MODEL.NAME]
    return proc