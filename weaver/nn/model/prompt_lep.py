import os
import torch
from torch import Tensor
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

Prompt_Lepton_Classifier = import_module(
    os.path.join(os.path.dirname(__file__), 'Prompt_Lepton_Classifier.py'), 'prompt_lep')

def get_model(data_config, **kwargs):

    ## number of classes
    num_classes = len(data_config.label_value)

    ## number of domain labels in the various regions (one binary or multiclass per region)
    num_domains = []
    if isinstance(data_config.label_domain_value, dict):
        for dct in data_config.label_domain_value.values():
            num_domains.append(len(dct))
    else:
        num_domains.append(len(data_config.label_domain_value))
   
    cfg = dict(
        pf_features_dims=len(data_config.input_dicts['pf_features']),
        sv_features_dims=len(data_config.input_dicts['sv_features']),
        lep_features_dims=len(data_config.input_dicts['lep_features']),
        num_classes=num_classes,
        num_domains=num_domains,
        hidden_dim=kwargs.get('hidden_dim', 128),
        dropout_rate=kwargs.get('dropout_rate', 0.1),
    )

    _logger.info('Model config: %s' % str(cfg))
    
    model = Prompt_Lepton_Classifier.SimpleParticleNet(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


class CrossEntropyLogCoshLossDomain(torch.nn.L1Loss):
    __constants__ = ['reduction','loss_lambda','loss_gamma','quantiles','loss_kappa','domain_weight','domain_dim']

    def __init__(self, 
                 reduction: str = 'mean', 
                 loss_kappa: float = 1., 
                 domain_weight: list = [],
                 domain_dim: list = [],
             ) -> None:
        super(CrossEntropyLogCoshLossDomain, self).__init__(None, None, reduction)
        self.loss_kappa = loss_kappa
        self.domain_weight = domain_weight
        self.domain_dim = domain_dim

    def forward(self, 
                input_cat: Tensor, y_cat: Tensor, 
                input_domain: Tensor, y_domain: Tensor, y_domain_check: Tensor) -> Tensor:
        
        # Classification term
        loss_cat  = 0
        if input_cat.nelement():
            loss_cat = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction)

        ## domain terms
        loss_domain    = 0
        if input_domain.nelement():
            ## just one domain region
            if not self.domain_weight or len(self.domain_weight) == 1:
                loss_domain = self.loss_kappa*torch.nn.functional.cross_entropy(input_domain,y_domain,reduction=self.reduction)
            else:
                ## more domain regions with different relative weights
                for id,w in enumerate(self.domain_weight):
                    id_dom  = id*self.domain_dim[id]
                    y_check = y_domain_check[:,id]
                    indexes = y_check.nonzero();                    
                    y_val   = input_domain[indexes,id_dom:id_dom+self.domain_dim[id]].squeeze()
                    y_pred  = y_domain[indexes,id].squeeze()
                    if y_val.nelement():
                        loss_domain += w*torch.nn.functional.cross_entropy(y_val,y_pred,reduction=self.reduction)
                loss_domain *= self.loss_kappa
            
        return loss_cat+loss_domain, loss_cat, loss_domain
    

def get_loss(data_config, **kwargs):

    ## number of domain regions
    wdomain = data_config.label_domain_loss_weight
    ## number of lables for cross entropy in each domain
    if type(data_config.label_domain_value) == dict:
        ldomain = [len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values()]
    else:
        ldomain = [len(data_config.label_domain_value)]

    return CrossEntropyLogCoshLossDomain(
        reduction=kwargs.get('reduction','mean'),
        loss_kappa=kwargs.get('loss_kappa',1),
        domain_weight=wdomain,
        domain_dim=ldomain
    )
