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
    def __init__(self, pf_features_dims, sv_features_dims, lep_features_dims, num_classes, hidden_dim=128, dropout_rate=0.1, for_inference=False, alpha_grad=1000.0):
        super(SimpleParticleNet, self).__init__()
        
        # Calculate the total input dimension
        input_dim = pf_features_dims * 50 + sv_features_dims * 5 + lep_features_dims * 1
        
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
        self.for_inference = for_inference

        # Classifier network
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, num_classes)
        # )
        
        # Regression network with Gradient Reversal
        # self.grl = GradientReversalLayer(lambda_=alpha_grad)
        # self.grl.register_full_backward_hook(full_backward_hook)
        # self.regressor = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, 1)
        # )
        self.reg_grl = GradientReversalLayer(lambda_=alpha_grad)
        # self.reg_grl.register_full_backward_hook(full_backward_hook)
        self.reg_fc1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.reg_fc2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.reg_fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.reg_fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.reg_fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.reg_fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.reg_fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.reg_fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.reg_fc9 = nn.Linear(hidden_dim, hidden_dim)
        self.reg_fc10 = nn.Linear(hidden_dim, 1)


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

        reg_x = self.reg_grl(x)

        # reg_x.register_hook(self.gradient_analysis_hook.after_reversal_hook)

        reg_x = F.relu(self.reg_fc1(reg_x))
        reg_x = self.dropout(reg_x)
        reg_x = F.relu(self.reg_fc2(reg_x))
        reg_x = self.dropout(reg_x)
        reg_x = F.relu(self.reg_fc3(reg_x))
        reg_x = self.dropout(reg_x)
        reg_x = F.relu(self.reg_fc4(reg_x))
        reg_x = self.dropout(reg_x)
        reg_x = F.relu(self.reg_fc5(reg_x))
        reg_x = self.dropout(reg_x)
        reg_x = F.relu(self.reg_fc6(reg_x))
        reg_x = self.dropout(reg_x)
        reg_x = F.relu(self.reg_fc7(reg_x))
        reg_x = self.dropout(reg_x)
        reg_x = F.relu(self.reg_fc8(reg_x))
        reg_x = self.dropout(reg_x)
        reg_x = F.relu(self.reg_fc9(reg_x))
        reg_x = self.dropout(reg_x)
        
        # domain_outputs = [reg_fc(reg_x)]
        reg_output = self.reg_fc10(reg_x)

        # # Print gradients
        # input_tensor, input_grad_before, input_grad_after = self.gradient_analysis_hook.get_gradients() 
        # print("Input Gradient Before Reversal:", input_grad_before)
        # print("Input Gradient After Reversal:", input_grad_after)

        if self.for_inference:
            # Apply softmax to class_output for inference
            class_output = torch.softmax(class_output, dim=1)
            # Linear activation function on reg_output = reg_output

        return torch.cat((class_output, reg_output), dim=1)