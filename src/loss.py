import torch
import torch.nn as nn
import torch.nn.functional as F

class KendallMultiTaskLoss(nn.Module):
    """
    Stage 6: Multi-task loss dynamically weighted by homoscedastic uncertainty
    (Kendall et al., 2018). It automatically learns to balance the disparate 
    magnitudes of coordinate regression vs classification task losses.
    """
    def __init__(self, num_tasks=5):
        super().__init__()
        # Initialize learnable variance weights to 0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, preds, targets):
        track_p, wind_p, ri_p, landfall_p = preds
        
        # Slicing targets based on LABEL_COLS design
        # 0:6 = Track (lat/lon), 6:9 = Wind speed, 9 = Saffir-Simpson Cat, 10 = RI, 11 = Landfall
        track_y = targets[:, 0:6]
        wind_spd_y = targets[:, 6:9]
        wind_cat_y = targets[:, 9].long()
        ri_y = targets[:, 10]
        
        # Landfall regression isn't natively in raw dataset yet, so we pad it
        # using the binary 72h landfall flag as a pseudo-target baseline
        landfall_y = torch.zeros_like(landfall_p) 
        landfall_y[:, 0] = targets[:, 11] 

        # Task 1: Track Error (MSE)
        L0 = F.mse_loss(track_p, track_y)
        
        # Task 2: Wind Speed Error (MSE)
        L1 = F.mse_loss(wind_p[:, :3], wind_spd_y)
        
        # Task 3: Saffir-Simpson Category (Cross Entropy)
        L2 = F.cross_entropy(wind_p[:, 3:], wind_cat_y)
        
        # Task 4: Rapid Intensification (BCE with Logits)
        L3 = F.binary_cross_entropy_with_logits(ri_p.squeeze(-1), ri_y)
        
        # Task 5: Landfall Location & Time (MSE)
        L4 = F.mse_loss(landfall_p, landfall_y)

        losses = [L0, L1, L2, L3, L4]
        total_loss = 0.0
        
        # Compute weighted sum: sum( 0.5 * exp(-log_var) * Loss + 0.5 * log_var )
        for i, l in enumerate(losses):    
            precision = torch.exp(-self.log_vars[i])
            total_loss += 0.5 * precision * l + 0.5 * self.log_vars[i]

        return total_loss, [l.detach().item() for l in losses]
