import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as all_gather_grad_safe
from einops import repeat, rearrange


class SigmoidAnnealing:
    """
    A sigmoid annealing function that gradually interpolates between start_value and end_value.

    Parameters:
        start_value (float): The initial value of the annealing process.
        end_value (float): The final value of the annealing process.
        step (int): The current step in the annealing process.
        total_steps (int): The total number of steps for annealing.

    Returns:
        float: The annealed value at the current step.
    """

    def __init__(
        self, init_beta, final_beta, start_annealing, end_annealing, steepness
    ):
        self.init_beta = init_beta
        self.final_beta = final_beta
        self.start_annealing = start_annealing
        self.end_annealing = end_annealing
        self.steepness = steepness
        self.midpoint = int(start_annealing + (end_annealing - start_annealing) / 2)

    def get_beta_value(self, step):
        if step < self.start_annealing:
            sigmoid = 1 / (1 + math.exp(self.steepness * self.midpoint))
            curr_beta = self.init_beta + (self.final_beta - self.init_beta) * sigmoid
        elif step >= self.start_annealing and step < self.end_annealing:
            sigmoid = 1 / (
                1
                + math.exp(
                    -self.steepness * ((step - self.start_annealing) - self.midpoint)
                )
            )
            curr_beta = self.init_beta + (self.final_beta - self.init_beta) * sigmoid
        else:
            sigmoid = 1 / (
                1
                + math.exp(
                    -self.steepness
                    * ((self.end_annealing - self.start_annealing) - self.midpoint)
                )
            )
            curr_beta = self.init_beta + (self.final_beta - self.init_beta) * sigmoid
        return curr_beta


class ExpAnnealing:
    """
    Compute temperature based on current step
    -> exponential temperature annealing
    """

    def __init__(self, init_beta, final_beta, start_annealing, end_annealing):
        self.init_beta = init_beta
        self.final_beta = final_beta
        self.start_annealing = start_annealing
        self.end_annealing = end_annealing
        self.rate = math.log(final_beta - init_beta + 1 + 1e-10) / float(
            end_annealing - start_annealing
        )

    def get_beta_value(self, step):
        if step < self.start_annealing:
            curr_beta = self.init_beta
        elif step >= self.start_annealing and step < self.end_annealing:
            curr_beta = min(
                (self.init_beta - 1) + math.exp(self.rate * step), self.final_beta
            )
        else:
            curr_beta = self.final_beta
        return curr_beta


class CosAnnealing:
    """
    Compute temperature based on current step
    -> cosine temperature annealing
    """

    def __init__(self, init_beta, final_beta, start_annealing, end_annealing):
        self.init_beta = init_beta
        self.final_beta = final_beta
        self.start_annealing = start_annealing
        self.end_annealing = end_annealing

    def get_beta_value(self, step):
        if step >= self.start_annealing and step < self.end_annealing:
            curr_beta = self.final_beta + 0.5 * (self.init_beta - self.final_beta) * (
                1
                + math.cos(
                    (step / (self.end_annealing - self.start_annealing)) * math.pi
                )
            )
        elif step < self.start_annealing:
            curr_beta = self.init_beta
        else:
            curr_beta = self.final_beta
        return curr_beta


class LinearAnnealing:
    def __init__(self, init_beta, final_beta, start_annealing, end_annealing):
        self.init_beta = init_beta
        self.final_beta = final_beta
        self.start_annealing = start_annealing
        self.end_annealing = end_annealing

    def get_beta_value(self, step):
        if step >= self.start_annealing and step < self.end_annealing:
            annealing_steps = self.end_annealing - self.start_annealing
            curr_beta = (1 - step / annealing_steps) * self.init_beta + (
                step / annealing_steps
            ) * self.final_beta
        elif step < self.start_annealing:
            curr_beta = self.init_beta
        else:
            curr_beta = self.final_beta
        return curr_beta


class NoAnnealing:
    def __init__(self, init_beta, final_beta, start_annealing, end_annealing):
        self.init_beta = init_beta
        self.final_beta = final_beta
        self.start_annealing = start_annealing
        self.end_annealing = end_annealing

    def get_beta_value(self, step):
        return self.final_beta


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def calc_similarity_batch(self, a, b):
        # a: [2B, D], b: [2N, D]
        return torch.matmul(a, rearrange(b, "b e -> e b"))  # [2B, 2N]
    
    def _gather_tensor(self, tensor):
        """
        Gathers a tensor from all processes while preserving the autograd graph.
        """
        if dist.is_available() and dist.is_initialized():
            tensor = all_gather_grad_safe(tensor)
            return torch.cat(tensor, dim=0)
        return tensor
    
    def forward(self, proj_1, proj_2, mod_mask_1, mod_mask_2, mm_mae_device):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        # Normalize projections
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        # Gather across all GPUs if using DDP
        z_i_all = self._gather_tensor(z_i)
        z_j_all = self._gather_tensor(z_j)
        z_all = torch.cat([z_i_all, z_j_all], dim=0)  # [2N, D]

        # Concatenate local batch to compute similarities to negatives
        z_combined = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        similarity_matrix = self.calc_similarity_batch(z_combined, z_all)  # [2B, 2N]
        
        # Local positives only
        positives = torch.sum(z_i * z_j, dim=1)  # [B]
        positives = torch.cat([positives, positives], dim=0)  # [2B]
        mod_mask_1_all = self._gather_tensor(mod_mask_1)
        mod_mask_2_all = self._gather_tensor(mod_mask_2)
        mod_masks_all = torch.cat(
            [torch.logical_and(mod_mask_1_all, mod_mask_2_all)] * 2, dim=0
        ) 
        # Build modality mask for negatives
        mod_masks_matrix = mod_masks_all.unsqueeze(0).expand(similarity_matrix.shape[0], -1)  # [2B, 2N]


        # Build index mask to remove self-comparisons
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_batch_size = proj_1.size(0)
            total_batch_size = z_all.size(0)
            # Compute which indices correspond to the local batch
            local_indices = torch.arange(2 * local_batch_size, device=mm_mae_device) + 2 * local_batch_size * rank
            all_indices = torch.arange(total_batch_size, device=mm_mae_device)
            idx_mask = (local_indices.unsqueeze(1) != all_indices.unsqueeze(0))  # [2B, 2N]
        else:
            # Single-device: mask self similarity diagonals
            total_batch_size = z_all.size(0)
            idx_mask = ~torch.eye(total_batch_size, device=mm_mae_device, dtype=torch.bool)
            idx_mask = idx_mask.expand(2 * proj_1.size(0), -1)

        # Final mask for denominator
        full_mask = idx_mask & mod_masks_matrix

        # Compute SimCLR-style contrastive loss, ensuring stability with epsilon, alternative: go back to previous version of masking out 0 denominator too
        nominator = torch.exp(positives / self.temperature)  # [2B]
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature) * full_mask, dim=1) + 1e-12  # [2B]
        all_losses = -torch.log(nominator / denominator)
        loss = all_losses.mean()
        return loss
