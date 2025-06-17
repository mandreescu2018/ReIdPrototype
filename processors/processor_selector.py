from .processor_standard import ProcessorStandard
from .processor_qaconv import ProcessorQaconv
from .processor_pat import ProcessorPat

__factory = {
    'vit_transformer': ProcessorStandard,
    'vit_transformer_vanilla': ProcessorStandard,
    'vit_transformer_jpm': ProcessorStandard,
    'vit_transformer_pytorch': ProcessorStandard,
    'mobilenet_v2': ProcessorStandard,
    'resnet50': ProcessorStandard,
    'simple_resnet50': ProcessorStandard,
    'hacnn': ProcessorStandard,
    'qaconv': ProcessorQaconv,
    'PAT': ProcessorPat,
    'vit_pat_transformer': ProcessorPat,

}

def get_processor(cfg):
    proc = __factory[cfg.MODEL.NAME]
    return proc