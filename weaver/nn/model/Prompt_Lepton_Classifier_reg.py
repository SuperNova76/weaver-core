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
    def __init__(self, pf_features_dims, sv_features_dims, lep_features_dims, num_classes, hidden_dim=256, dropout_rate=0.1, for_inference=False, alpha_grad=0.0):
        super(SimpleParticleNet, self).__init__()
        
        # Calculate the total input dimension
        input_dim = pf_features_dims * 50 + sv_features_dims * 5 + lep_features_dims * 1
        
        # Shared network layers
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

        # Classifier network
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
        
        # Regression network with Gradient Reversal
        self.grl = GradientReversalLayer(lambda_=alpha_grad)
        # self.grl.register_full_backward_hook(full_backward_hook)
        self.regressor = nn.Sequential(
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
            nn.Linear(hidden_dim, 1)
        )

        self.num_classes = num_classes
        self.for_inference = for_inference

    #     # Placeholder to store gradients for printing
    #     self.gradients_before = None
    #     self.gradients_after = None
    #     self.register_hooks()

    # def register_hooks(self):
    #     """Register hooks to capture gradients before and after the gradient reversal layer."""
    #     def save_grad_before_grl(grad):
    #         self.gradients_before = grad

    #     def save_grad_after_grl(grad):
    #         self.gradients_after = grad

    #     # Hook for gradients before gradient reversal
    #     self.grl.register_backward_hook(
    #         lambda module, grad_input, grad_output: save_grad_before_grl(grad_input[0])
    #     )

    #     # Hook for gradients after gradient reversal
    #     self.regressor[0].register_backward_hook(
    #         lambda module, grad_input, grad_output: save_grad_after_grl(grad_output[0])
    #     )

    def forward(self, pf_features, sv_features, lep_features, *args):
        # Flatten the inputs
        pf_features = pf_features.view(pf_features.size(0), -1)
        sv_features = sv_features.view(sv_features.size(0), -1)
        lep_features = lep_features.view(lep_features.size(0), -1)
        
        # Concatenate flattened inputs
        x = torch.cat((pf_features, sv_features, lep_features), dim=1)
        
        # Forward pass through shared layers
        shared_features = self.shared_layers(x)
        
        # Forward pass through the classifier
        class_output = self.classifier(shared_features)
        
        # Forward pass through the gradient reversal layer
        reversed_features = self.grl(shared_features)
        
        # Forward pass through the regressor
        reg_output = self.regressor(reversed_features)

        if self.for_inference:
            # Apply softmax to class_output for inference
            class_output = torch.softmax(class_output, dim=1)

        return torch.cat((class_output, reg_output), dim=1)