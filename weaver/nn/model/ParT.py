import os
import torch
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParT = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleTransformer.py'), 'ParT')


def get_model(data_config, **kwargs):

    cfg = dict(
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        remove_self_pair=False,
        use_pre_norm_pair=False,
        embed_dims=[128, 128, 128],
        # embed_dims=[64, 64, 64],
        # embed_dims=[32, 32, 32],
        # embed_dims=[32, 32],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        # cls_block_params={'dropout': 0.1, 'attn_dropout': 0.1, 'activation_dropout': 0.1},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )

    # if 'lep_features' in data_config.input_dicts:
        # 1L and 2L
    cfg.update(
        pf_input_dim=len(data_config.input_dicts['pf_features']),
        sv_input_dim=len(data_config.input_dicts['sv_features']),
        lep_input_dim=len(data_config.input_dicts['lep_features']),
    )
    ParticleTransformer = ParT.ParticleTransformerTagger
    # else:
    #     # 0L
    #     cfg.update(
    #         input_dim=len(data_config.input_dicts['pf_features']),
    #     )
    #     ParticleTransformer = ParT.ParticleTransformer

    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))
    
    model = ParticleTransformer(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
