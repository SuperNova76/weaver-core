import os
import torch
from torch import Tensor
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module
import math

Prompt_Lepton_Classifier = import_module(
    os.path.join(os.path.dirname(__file__), 'Prompt_Lepton_Classifier_reg_v2.py'), 'prompt_lep')

def get_model(data_config, **kwargs):

    ## number of classes
    num_classes = len(data_config.label_value)

    ## number of domain labels in the various regions (one binary or multiclass per region)
    # num_domains = []
    # if isinstance(data_config.label_domain_value, dict):
    #     for dct in data_config.label_domain_value.values():
    #         num_domains.append(len(dct))
    # else:
    #     num_domains.append(len(data_config.label_domain_value))
   
    cfg = dict(
        pf_features_dims=len(data_config.input_dicts['pf_features']),
        sv_features_dims=len(data_config.input_dicts['sv_features']),
        lep_features_dims=len(data_config.input_dicts['lep_features']),
        num_classes=num_classes,
        # num_domains=num_domains,
        hidden_dim=kwargs.get('hidden_dim', 128),
        dropout_rate=kwargs.get('dropout_rate', 0.1),
        for_inference=kwargs.get('for_inference', False),  # Ensure for_inference is passed here
        alpha_grad=kwargs.get('alpha_grad', 1000.0),  # Ensure alpha_grad is passed here
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


class CrossEntropyLogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction','nclass','ntarget','loss_lambda']

    def __init__(self, reduction: str = 'mean', nclass: int = 1, ntarget: int = 1, loss_lambda: float = 1.) -> None:
        super(CrossEntropyLogCoshLoss, self).__init__(None, None, reduction)
        self.nclass = nclass;
        self.ntarget = ntarget;
        self.loss_lambda = loss_lambda

    def forward(self, input: Tensor, y_cat: Tensor, inputReg: Tensor, y_reg: Tensor) -> Tensor:

        ## regression term
        # input_reg = input[:,self.nclass:self.nclass+self.ntarget].squeeze();
        input_reg = inputReg.squeeze();
        y_reg     = y_reg.squeeze();
        # import pdb; pdb.set_trace()
        # loss_reg  = (input_reg-y_reg)+torch.nn.functional.softplus(-2.*(input_reg-y_reg))-math.log(2);
        loss_reg = torch.nn.functional.mse_loss(inputReg.squeeze(), y_reg.squeeze());
        ## classification term
        input_cat = input[:,:self.nclass].squeeze();
        y_cat     = y_cat.squeeze().long();
        loss_cat  = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);
                
        # print ("input_reg")
        # print (input_reg)
        # print ("y_reg")
        # print (y_reg)

        ## final loss and pooling over batcc
        if self.reduction == 'none':            
            return loss_cat+self.loss_lambda*loss_reg, loss_cat, loss_reg*self.loss_lambda;
        elif self.reduction == 'mean':
            return loss_cat+self.loss_lambda*loss_reg.mean(), loss_cat, loss_reg.mean()*self.loss_lambda;
        elif self.reduction == 'sum':
            return loss_cat+self.loss_lambda*loss_reg.sum(), loss_cat, loss_reg.sum()*self.loss_lambda;


def get_loss(data_config, **kwargs):
    nclass  = len(data_config.label_value);
    ntarget = len(data_config.target_value);
    return CrossEntropyLogCoshLoss(reduction=kwargs.get('reduction','mean'),loss_lambda=kwargs.get('loss_lambda',1),nclass=nclass,ntarget=ntarget);
