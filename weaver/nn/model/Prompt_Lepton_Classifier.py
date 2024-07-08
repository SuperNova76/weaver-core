import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleParticleNet(nn.Module):
    def __init__(self, pf_features_dims, sv_features_dims, lep_features_dims, num_classes, num_domains, hidden_dim=128, dropout_rate=0.1):
        super(SimpleParticleNet, self).__init__()
        
        # Calculate the total input dimension
        input_dim = pf_features_dims * 50 + sv_features_dims * 5 + lep_features_dims * 1  # Adjust according to the lengths defined in YAML
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        # Domain classification layers
        self.domain_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.domain_fc3 = nn.ModuleList([nn.Linear(hidden_dim, num_domain) for num_domain in num_domains])
        
    def forward(self, pf_features, sv_features, lep_features, *args):
        # Flatten the inputs
        pf_features = pf_features.view(pf_features.size(0), -1)
        sv_features = sv_features.view(sv_features.size(0), -1)
        lep_features = lep_features.view(lep_features.size(0), -1)
        
        # Concatenate flattened inputs
        x = torch.cat((pf_features, sv_features, lep_features), dim=1)
        
        # Forward pass through fully connected layers with ReLU activations and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Classification output
        class_output = self.fc3(x)

        # Domain classification output
        domain_outputs = []
        domain_x = F.relu(self.domain_fc1(x))
        domain_x = self.dropout(domain_x)
        domain_x = F.relu(self.domain_fc2(domain_x))
        domain_x = self.dropout(domain_x)
        for fc in self.domain_fc3:
            domain_outputs.append(fc(domain_x))

        return class_output, domain_outputs
