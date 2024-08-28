import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer"""

    @staticmethod
    def forward(ctx, x, lambda_):
        """Forward pass"""
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass"""
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer Module"""

    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)

class SimpleParticleNet(nn.Module):
    def __init__(self, pf_features_dims, sv_features_dims, lep_features_dims, num_classes, num_domains, hidden_dim=128, dropout_rate=0.1, for_inference=False, alpha_grad=1000.0):
        super(SimpleParticleNet, self).__init__()
        
        # Calculate the total input dimension
        input_dim = pf_features_dims * 50 + sv_features_dims * 5 + lep_features_dims * 1  # Adjust according to the lengths defined in YAML
        self.fc1 = nn.Linear(input_dim, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.num_classes = num_classes
        self.num_domains = len(num_domains)
        self.for_inference = for_inference

        # Domain classification layers with gradient reversal
        self.domain_grl = GradientReversalLayer(lambda_=alpha_grad)
        self.domain_fc1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.domain_fc2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.domain_fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.domain_fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc9 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc10 = nn.ModuleList([nn.Linear(hidden_dim, num_domain) for num_domain in num_domains])

        # self.gradient_analysis_hook = GradientAnalysisHook()
        # self.hook_handle = None
        
    def forward(self, pf_features, sv_features, lep_features, *args):
        # Flatten the inputs
        pf_features = pf_features.view(pf_features.size(0), -1)
        sv_features = sv_features.view(sv_features.size(0), -1)
        lep_features = lep_features.view(lep_features.size(0), -1)
        
        # Concatenate flattened inputs
        x = torch.cat((pf_features, sv_features, lep_features), dim=1)
        
        # Forward pass through fully connected layers with ReLU activations and dropout
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = F.gelu(self.fc3(x))
        x = self.dropout(x)
        x = F.gelu(self.fc4(x))
        x = self.dropout(x)
        x = F.gelu(self.fc5(x))
        x = self.dropout(x)
        x = F.gelu(self.fc6(x))
        x = self.dropout(x)
        x = F.gelu(self.fc7(x))
        x = self.dropout(x)

        # Classification output
        class_output = self.fc8(x)

        if self.num_domains > 0:
            # x.requires_grad_(True)
            # if self.gradient_analysis_hook.input_tensor is None:
            #     self.gradient_analysis_hook.input_tensor = x.clone().detach()
            #     # x.register_hook(self.gradient_analysis_hook.before_reversal_hook)

            domain_x = self.domain_grl(x)
            # Ensure domain_x retains requires_grad=True
            domain_x = domain_x.detach().requires_grad_()

            # domain_x.register_hook(self.gradient_analysis_hook.after_reversal_hook)

            domain_x = F.gelu(self.domain_fc1(domain_x))
            domain_x = self.dropout(domain_x)
            domain_x = F.gelu(self.domain_fc2(domain_x))
            domain_x = self.dropout(domain_x)
            domain_x = F.gelu(self.domain_fc3(domain_x))
            domain_x = self.dropout(domain_x)
            domain_x = F.gelu(self.domain_fc4(domain_x))
            domain_x = self.dropout(domain_x)
            domain_x = F.gelu(self.domain_fc5(domain_x))
            domain_x = self.dropout(domain_x)
            domain_x = F.gelu(self.domain_fc6(domain_x))
            domain_x = self.dropout(domain_x)
            domain_x = F.gelu(self.domain_fc7(domain_x))
            domain_x = self.dropout(domain_x)
            domain_x = F.gelu(self.domain_fc8(domain_x))
            domain_x = self.dropout(domain_x)
            domain_x = F.gelu(self.domain_fc9(domain_x))
            domain_x = self.dropout(domain_x)
            
            domain_outputs = [domain_fc(domain_x) for domain_fc in self.domain_fc10]
            domain_outputs = torch.cat(domain_outputs, dim=1)

            # # Print gradients
            # input_tensor, input_grad_before, input_grad_after = self.gradient_analysis_hook.get_gradients() 
            # print("Input Gradient Before Reversal:", input_grad_before)
            # print("Input Gradient After Reversal:", input_grad_after)

        if self.for_inference:
            # Apply softmax to class_output
            class_output = torch.softmax(class_output, dim=1)
            
            if self.num_domains > 0:
                # Apply softmax to each domain_output
                domain_outputs = torch.cat([torch.softmax(domain_output, dim=1) for domain_output in domain_outputs.chunk(len(self.num_domains), dim=1)], dim=1)
        
        if self.num_domains > 0:
            return torch.cat((class_output, domain_outputs), dim=1)
        else:
            return class_output