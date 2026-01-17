"""
Simplified ANFIS Model for Rating Prediction
"""

import torch
import torch.nn as nn
import numpy as np


class SimplifiedANFIS(nn.Module):
    """
    Simplified Adaptive Neuro-Fuzzy Inference System

    Architecture:
    - Layer 1: Fuzzification (Gaussian membership functions)
    - Layer 2: Fuzzy rules (product t-norm)
    - Layer 3: Normalization
    - Layer 4: Defuzzification (weighted linear combination)
    """

    def __init__(self, n_inputs=8, n_rules=16, n_mfs=3):
        """
        Args:
            n_inputs: Number of input features (now 8)
            n_rules: Number of fuzzy rules
            n_mfs: Number of membership functions per input
        """
        super(SimplifiedANFIS, self).__init__()

        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.n_mfs = n_mfs

        # Layer 1: Membership function parameters (mean and std)
        # Each input has n_mfs membership functions
        self.mf_means = nn.Parameter(torch.randn(n_inputs, n_mfs))
        self.mf_stds = nn.Parameter(torch.ones(n_inputs, n_mfs) * 0.3)

        # Layer 2-3: Rule weights (learnable)
        self.rule_weights = nn.Parameter(torch.randn(n_rules, n_inputs * n_mfs))

        # Layer 4: Consequent parameters (linear combination per rule)
        self.consequent_params = nn.Parameter(torch.randn(n_rules, n_inputs + 1))

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameters with reasonable values"""
        # Initialize membership function means evenly across [0, 1]
        with torch.no_grad():
            for i in range(self.n_inputs):
                self.mf_means[i] = torch.linspace(0, 1, self.n_mfs)

            # Initialize stds
            self.mf_stds.data.fill_(0.3)

            # Initialize rule weights with Xavier
            nn.init.xavier_uniform_(self.rule_weights)

            # Initialize consequent parameters small
            self.consequent_params.data.uniform_(-0.1, 0.1)

    def gaussian_mf(self, x, mean, std):
        """Gaussian membership function"""
        return torch.exp(-((x - mean) ** 2) / (2 * std ** 2 + 1e-8))

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, n_inputs)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.size(0)

        # Layer 1: Fuzzification
        # Compute membership degrees for each input
        mf_outputs = []
        for i in range(self.n_inputs):
            input_val = x[:, i].unsqueeze(1)  # (batch_size, 1)
            means = self.mf_means[i].unsqueeze(0)  # (1, n_mfs)
            stds = torch.clamp(self.mf_stds[i].unsqueeze(0), min=0.01)  # (1, n_mfs)

            mf_out = self.gaussian_mf(input_val, means, stds)  # (batch_size, n_mfs)
            mf_outputs.append(mf_out)

        # Concatenate all membership degrees
        all_mfs = torch.cat(mf_outputs, dim=1)  # (batch_size, n_inputs * n_mfs)

        # Layer 2: Fuzzy rules (weighted combination of membership functions)
        rule_activations = torch.mm(all_mfs, self.rule_weights.t())  # (batch_size, n_rules)
        rule_activations = torch.sigmoid(rule_activations)  # Normalize to [0, 1]

        # Layer 3: Normalize rule activations
        rule_sum = rule_activations.sum(dim=1, keepdim=True) + 1e-8
        normalized_rules = rule_activations / rule_sum  # (batch_size, n_rules)

        # Layer 4: Defuzzification (weighted linear combination)
        # Add bias term to input
        x_with_bias = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)

        # Compute consequent for each rule
        consequents = torch.mm(x_with_bias, self.consequent_params.t())  # (batch_size, n_rules)

        # Weighted sum based on rule activations
        output = (normalized_rules * consequents).sum(dim=1, keepdim=True)

        # Scale output to rating range [1, 5]
        output = torch.sigmoid(output) * 4 + 1

        return output.squeeze(1)

    def get_rules_summary(self):
        """Extract learned fuzzy rules in human-readable format"""
        rules = []

        with torch.no_grad():
            for rule_idx in range(self.n_rules):
                params = self.consequent_params[rule_idx].cpu().numpy()

                # Create rule formula
                terms = []
                for i in range(self.n_inputs):
                    coef = params[i]
                    if abs(coef) > 0.01:  # Only include significant terms
                        feature_names = [
                            'activity', 'popularity', 'user_avg', 'movie_avg',
                            'user_std', 'genre_affinity', 'user_dev', 'movie_dev'
                        ]
                        terms.append(f"{coef:.3f}*{feature_names[i]}")

                bias = params[-1]
                formula = f"rating = {bias:.3f}"
                if terms:
                    formula += " + " + " + ".join(terms)

                rules.append({
                    'rule_id': rule_idx,
                    'formula': formula,
                    'coefficients': params.tolist()
                })

        return rules

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)