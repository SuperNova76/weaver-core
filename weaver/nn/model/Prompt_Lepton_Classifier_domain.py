import torch
import torch.nn as nn
import torch.nn.functional as F

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

def full_backward_hook(module, grad_input, grad_output):
    print("Gradient before reversal:", grad_input[0])
    print("Gradient after reversal:", grad_output[0])

class SimpleParticleNet(nn.Module):
    def __init__(self, pf_features_dims, sv_features_dims, lep_features_dims, num_classes, num_domains, hidden_dim=256, dropout_rate=0.1, for_inference=False, alpha_grad=0.7):
        super(SimpleParticleNet, self).__init__()

        self.dropout_rate = dropout_rate

        input_dim = pf_features_dims * 50 + sv_features_dims * 5 + lep_features_dims * 1
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        self.num_classes = num_classes
        self.num_domains = len(num_domains)
        self.for_inference = for_inference

        # self.alpha_grad = alpha_grad

        # Regression network with Gradient Reversal
        self.grl = GradientReversalLayer(lambda_=alpha_grad)
        # self.grl.register_full_backward_hook(full_backward_hook)

        self.domain_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc5 = nn.ModuleList([nn.Linear(hidden_dim, num_domain) for num_domain in num_domains])

        

    def forward(self, pf_features, sv_features, lep_features, *args):
        # Flatten the inputs
        pf_features = pf_features.view(pf_features.size(0), -1)
        sv_features = sv_features.view(sv_features.size(0), -1)
        lep_features = lep_features.view(lep_features.size(0), -1)

        # Concatenate flattened inputs
        x = torch.cat((pf_features, sv_features, lep_features), dim=1)

        shared_features = self.shared_layers(x)
        class_output = self.classifier(shared_features)

        if self.num_domains > 0:
            # Apply gradient reversal
            reversed_features = self.grl(shared_features)

            domain_x = F.relu(self.domain_fc1(reversed_features))
            domain_x = F.dropout(domain_x, p=self.dropout_rate)  # Apply dropout here
            domain_x = F.relu(self.domain_fc2(domain_x))
            domain_x = F.dropout(domain_x, p=self.dropout_rate)  # Apply dropout here
            domain_x = F.relu(self.domain_fc3(domain_x))
            domain_x = F.dropout(domain_x, p=self.dropout_rate)
            domain_x = F.relu(self.domain_fc4(domain_x))
            domain_x = F.dropout(domain_x, p=self.dropout_rate)

            domain_outputs = [domain_fc(domain_x) for domain_fc in self.domain_fc5]
            domain_outputs = torch.cat(domain_outputs, dim=1)

        if self.for_inference:
            # Apply softmax to class_output
            class_output = torch.softmax(class_output, dim=1)

            if self.num_domains > 0:
                # Apply softmax to each domain_output
                domain_outputs = torch.cat([torch.softmax(domain_output, dim=1) for domain_output in domain_outputs.chunk(self.num_domains, dim=1)], dim=1)

        if self.num_domains > 0:
            return torch.cat((class_output, domain_outputs), dim=1)
        else:
            return class_output
