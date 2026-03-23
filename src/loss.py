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
        wind_cat_y = targets[:, 9] # Keep float to scan for NaNs
        ri_y = targets[:, 10]
        
        # Landfall regression isn't natively in raw dataset yet, so we pad it
        # using the binary 72h landfall flag as a pseudo-target baseline
        landfall_y = torch.zeros_like(landfall_p) 
        landfall_y[:, 0] = targets[:, 11] 

        # Filter out NaN labels via PyTorch Tensor Masking
        v_track = ~torch.isnan(track_y).any(dim=1)
        v_wind  = ~torch.isnan(wind_spd_y).any(dim=1)
        v_cat   = ~torch.isnan(wind_cat_y)
        v_ri    = ~torch.isnan(ri_y)
        v_lf    = ~torch.isnan(landfall_y[:, 0])

        dev = preds[0].device
        # Task 1: Track Error (MSE)
        L0 = F.mse_loss(track_p[v_track], track_y[v_track]) if v_track.sum() > 0 else torch.tensor(0.0, device=dev)
        
        # Task 2: Wind Speed Error (MSE)
        L1 = F.mse_loss(wind_p[v_wind, :3], wind_spd_y[v_wind]) if v_wind.sum() > 0 else torch.tensor(0.0, device=dev)
        
        # Task 3: Saffir-Simpson Category (Cross Entropy)
        L2 = F.cross_entropy(wind_p[v_cat, 3:], wind_cat_y[v_cat].long()) if v_cat.sum() > 0 else torch.tensor(0.0, device=dev)
        
        # Task 4: Rapid Intensification (BCE with Logits)
        L3 = F.binary_cross_entropy_with_logits(ri_p[v_ri].squeeze(-1), ri_y[v_ri]) if v_ri.sum() > 0 else torch.tensor(0.0, device=dev)
        
        # Task 5: Landfall Location & Time (MSE)
        L4 = F.mse_loss(landfall_p[v_lf], landfall_y[v_lf]) if v_lf.sum() > 0 else torch.tensor(0.0, device=dev)

        losses = [L0, L1, L2, L3, L4]
        total_loss = 0.0
        
        # Compute weighted sum: sum( 0.5 * exp(-log_var) * Loss + 0.5 * log_var )
        for i, l in enumerate(losses):    
            precision = torch.exp(-self.log_vars[i])
            total_loss += 0.5 * precision * l + 0.5 * self.log_vars[i]

        return total_loss, [l.detach().item() for l in losses]
