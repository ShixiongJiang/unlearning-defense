import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np

# ============================================================================
# Feature Encoders
# ============================================================================

class FeatureEncoder(nn.Module):
    """Learn task-relevant representations for meaningful distance metrics"""
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.latent_dim = latent_dim
        
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.encoder(x)


class TrajectoryEncoder(nn.Module):
    """Encode entire trajectories for trajectory-level matching"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.hidden_dim = hidden_dim
        
    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj: (batch, seq_len, state_dim + action_dim)
        Returns:
            encoding: (batch, hidden_dim)
        """
        _, hidden = self.gru(traj)  # hidden: (num_layers, batch, hidden_dim)
        return hidden[-1]  # Return last layer


class EnsembleCritic(nn.Module):
    """Ensemble of critics for uncertainty estimation"""
    def __init__(self, state_dim: int, action_dim: int, num_ensemble: int = 5):
        super().__init__()
        self.ensemble = nn.ModuleList([
            self._make_critic(state_dim, action_dim) 
            for _ in range(num_ensemble)
        ])
        self.num_ensemble = num_ensemble
    
    def _make_critic(self, state_dim: int, action_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, obs: torch.Tensor, act: torch.Tensor, 
                return_all: bool = False) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        outputs = torch.stack([critic(x) for critic in self.ensemble], dim=0)
        
        if return_all:
            return outputs  # (num_ensemble, batch, 1)
        else:
            return outputs.mean(dim=0)  # (batch, 1)
    
    def get_uncertainty(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Compute epistemic uncertainty (ensemble variance)"""
        outputs = self.forward(obs, act, return_all=True)
        return outputs.var(dim=0)  # (batch, 1)


# ============================================================================
# Relabeling Strategies
# ============================================================================

class RelabelingStrategy:
    """Base class for different relabeling strategies"""
    
    @staticmethod
    def compute_pairwise_distances(
        feat_f: torch.Tensor, 
        feat_k: torch.Tensor,
        metric: str = 'euclidean'
    ) -> torch.Tensor:
        """
        Compute pairwise distances between forget and keep features
        Args:
            feat_f: (batch_f, feat_dim)
            feat_k: (batch_k, feat_dim)
        Returns:
            distances: (batch_f, batch_k)
        """
        if metric == 'euclidean':
            return torch.cdist(feat_f, feat_k, p=2)
        elif metric == 'cosine':
            feat_f_norm = F.normalize(feat_f, dim=-1)
            feat_k_norm = F.normalize(feat_k, dim=-1)
            return 1 - torch.mm(feat_f_norm, feat_k_norm.t())
        else:
            raise ValueError(f"Unknown metric: {metric}")


class KNNRelabeling(RelabelingStrategy):
    """K-Nearest Neighbors with weighted averaging"""
    
    def __init__(self, k: int = 5, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature
    
    def relabel(
        self, 
        batch_forget: Tuple, 
        batch_keep: Tuple,
        feature_encoder: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_forget: (obs_f, next_obs_f, act_f, rew_f, cost_f, done_f)
            batch_keep: (obs_k, next_obs_k, act_k, rew_k, cost_k, done_k)
            feature_encoder: Optional learned feature encoder
        Returns:
            rew_f_relabeled, cost_f_relabeled
        """
        obs_f, _, act_f, rew_f, cost_f, _ = batch_forget
        obs_k, _, act_k, rew_k, cost_k, _ = batch_keep
        
        # Compute features
        if feature_encoder is not None:
            with torch.no_grad():
                feat_f = feature_encoder(obs_f, act_f)
                feat_k = feature_encoder(obs_k, act_k)
        else:
            feat_f = torch.cat([obs_f, act_f], dim=-1)
            feat_k = torch.cat([obs_k, act_k], dim=-1)
        
        # Compute distances
        distances = self.compute_pairwise_distances(feat_f, feat_k)
        
        # Get k-nearest neighbors
        k = min(self.k, feat_k.shape[0])  # Handle small batches
        topk_dists, topk_idx = torch.topk(
            distances, k, dim=1, largest=False
        )  # (batch_f, k)
        
        # Compute softmax weights (inverse distance)
        weights = torch.softmax(-topk_dists / self.temperature, dim=1)
        
        # Gather neighbor values
        rew_neighbors = rew_k[topk_idx]  # (batch_f, k)
        cost_neighbors = cost_k[topk_idx]
        
        # Weighted average
        rew_f_relabeled = (weights * rew_neighbors).sum(dim=1, keepdim=True)
        cost_f_relabeled = (weights * cost_neighbors).sum(dim=1, keepdim=True)
        
        return rew_f_relabeled, cost_f_relabeled


class ConfidenceWeightedRelabeling(RelabelingStrategy):
    """Relabeling weighted by value function confidence"""
    
    def __init__(self, k: int = 5, temp_dist: float = 1.0, temp_conf: float = 1.0):
        self.k = k
        self.temp_dist = temp_dist
        self.temp_conf = temp_conf
    
    def relabel(
        self,
        batch_forget: Tuple,
        batch_keep: Tuple,
        feature_encoder: Optional[nn.Module],
        q_ensemble: Optional[EnsembleCritic] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_forget: (obs_f, next_obs_f, act_f, rew_f, cost_f, done_f)
            batch_keep: (obs_k, next_obs_k, act_k, rew_k, cost_k, done_k)
            feature_encoder: Learned feature encoder
            q_ensemble: Ensemble critic for uncertainty estimation
        Returns:
            rew_f_relabeled, cost_f_relabeled
        """
        obs_f, _, act_f, rew_f, cost_f, _ = batch_forget
        obs_k, _, act_k, rew_k, cost_k, _ = batch_keep
        
        # Compute features
        if feature_encoder is not None:
            with torch.no_grad():
                feat_f = feature_encoder(obs_f, act_f)
                feat_k = feature_encoder(obs_k, act_k)
        else:
            feat_f = torch.cat([obs_f, act_f], dim=-1)
            feat_k = torch.cat([obs_k, act_k], dim=-1)
        
        # Compute distances
        distances = self.compute_pairwise_distances(feat_f, feat_k)
        
        # Get k-nearest neighbors
        k = min(self.k, feat_k.shape[0])
        topk_dists, topk_idx = torch.topk(
            distances, k, dim=1, largest=False
        )
        
        # Distance-based weights
        distance_weights = torch.softmax(-topk_dists / self.temp_dist, dim=1)
        
        # Confidence-based weights (if ensemble available)
        if q_ensemble is not None:
            with torch.no_grad():
                # Get uncertainty for keep neighbors
                obs_k_neighbors = obs_k[topk_idx]  # (batch_f, k, obs_dim)
                act_k_neighbors = act_k[topk_idx]
                
                batch_f, k_val = obs_k_neighbors.shape[:2]
                obs_k_flat = obs_k_neighbors.reshape(-1, obs_k_neighbors.shape[-1])
                act_k_flat = act_k_neighbors.reshape(-1, act_k_neighbors.shape[-1])
                
                uncertainty = q_ensemble.get_uncertainty(obs_k_flat, act_k_flat)
                uncertainty = uncertainty.reshape(batch_f, k_val)
                
                # Confidence = 1 / (1 + uncertainty)
                confidence = 1.0 / (1.0 + uncertainty.squeeze(-1))
                confidence_weights = torch.softmax(confidence / self.temp_conf, dim=1)
        else:
            confidence_weights = torch.ones_like(distance_weights)
        
        # Combined weights
        weights = distance_weights * confidence_weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Gather and weight
        rew_neighbors = rew_k[topk_idx]
        cost_neighbors = cost_k[topk_idx]
        
        rew_f_relabeled = (weights * rew_neighbors).sum(dim=1, keepdim=True)
        cost_f_relabeled = (weights * cost_neighbors).sum(dim=1, keepdim=True)
        
        return rew_f_relabeled, cost_f_relabeled


class TrajectoryRelabeling(RelabelingStrategy):
    """Match entire trajectories for temporal coherence"""
    
    def __init__(self, traj_encoder: TrajectoryEncoder):
        self.traj_encoder = traj_encoder
    
    def relabel_trajectories(
        self,
        trajs_forget: List[torch.Tensor],
        trajs_keep: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            trajs_forget: List of trajectories (T, state_dim + action_dim)
            trajs_keep: List of trajectories
        Returns:
            trajs_forget_relabeled: Forget trajs with relabeled rewards/costs
        """
        # Pad trajectories to same length
        def pad_trajs(trajs):
            max_len = max(t.shape[0] for t in trajs)
            padded = []
            for traj in trajs:
                if traj.shape[0] < max_len:
                    padding = torch.zeros(
                        max_len - traj.shape[0], traj.shape[1],
                        device=traj.device, dtype=traj.dtype
                    )
                    traj = torch.cat([traj, padding], dim=0)
                padded.append(traj)
            return torch.stack(padded)
        
        trajs_f_padded = pad_trajs(trajs_forget)  # (num_f, max_len, dim)
        trajs_k_padded = pad_trajs(trajs_keep)
        
        with torch.no_grad():
            # Encode trajectories
            feat_f = self.traj_encoder(trajs_f_padded)  # (num_f, hidden_dim)
            feat_k = self.traj_encoder(trajs_k_padded)
        
        # Match trajectories
        distances = self.compute_pairwise_distances(feat_f, feat_k)
        nearest_idx = distances.argmin(dim=1)
        
        # Relabel (simplified - assumes trajectories have reward/cost fields)
        trajs_relabeled = []
        for i, traj_f in enumerate(trajs_forget):
            matched_traj_k = trajs_keep[nearest_idx[i]]
            # This is a simplified version - you'd need to properly handle
            # your trajectory structure
            trajs_relabeled.append({
                'obs': traj_f['obs'],
                'act': traj_f['act'],
                'rew': matched_traj_k['rew'][:len(traj_f['rew'])],
                'cost': matched_traj_k['cost'][:len(traj_f['cost'])],
            })
        
        return trajs_relabeled


# ============================================================================
# Main Unlearning Class
# ============================================================================

class RelabelingUnlearner:
    """Complete relabeling-based unlearning system"""
    
    def __init__(
        self,
        model,  # Your existing safe RL model
        state_dim: int,
        action_dim: int,
        device: str = 'cuda',
        # Feature encoder params
        use_feature_encoder: bool = True,
        feature_latent_dim: int = 128,
        # Relabeling params
        relabel_strategy: str = 'knn',  # 'knn', 'confidence', 'trajectory'
        k_neighbors: int = 5,
        temperature: float = 1.0,
        # Confidence params
        use_confidence_weighting: bool = False,
        num_ensemble: int = 5,
        # Gradual relabeling params
        use_gradual_relabeling: bool = True,
        gradual_steps: int = 1000,
        # Training params
        encoder_lr: float = 3e-4,
        encoder_update_freq: int = 1,
    ):
        self.model = model
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Feature encoder
        self.use_feature_encoder = use_feature_encoder
        if use_feature_encoder:
            self.feature_encoder = FeatureEncoder(
                state_dim, action_dim, feature_latent_dim
            ).to(device)
            self.encoder_optim = torch.optim.Adam(
                self.feature_encoder.parameters(), lr=encoder_lr
            )
        else:
            self.feature_encoder = None
        
        # Ensemble for confidence (optional)
        self.use_confidence_weighting = use_confidence_weighting
        if use_confidence_weighting:
            self.q_ensemble = EnsembleCritic(
                state_dim, action_dim, num_ensemble
            ).to(device)
            self.ensemble_optim = torch.optim.Adam(
                self.q_ensemble.parameters(), lr=3e-4
            )
        else:
            self.q_ensemble = None
        
        # Relabeling strategy
        self.relabel_strategy = relabel_strategy
        if relabel_strategy == 'knn':
            self.relabeler = KNNRelabeling(k=k_neighbors, temperature=temperature)
        elif relabel_strategy == 'confidence':
            self.relabeler = ConfidenceWeightedRelabeling(
                k=k_neighbors, temp_dist=temperature, temp_conf=1.0
            )
        else:
            raise ValueError(f"Unknown strategy: {relabel_strategy}")
        
        # Gradual relabeling
        self.use_gradual_relabeling = use_gradual_relabeling
        self.gradual_steps = gradual_steps
        self.current_step = 0
        
        # Counters
        self.encoder_update_freq = encoder_update_freq
        self.update_counter = 0
    
    def train_feature_encoder(
        self, 
        batch_keep: Tuple,
        temperature: float = 0.5
    ) -> float:
        """
        Train feature encoder with contrastive learning on keep data
        Temporal coherence: nearby states should have similar features
        """
        if not self.use_feature_encoder:
            return 0.0
        
        obs_k, next_obs_k, act_k, _, _, done_k = batch_keep
        
        # Encode current state-actions
        feat_current = self.feature_encoder(obs_k, act_k)
        
        # Positive pairs: temporally adjacent (current → next)
        # For next actions, we can use the policy or sampled actions
        with torch.no_grad():
            # Sample next actions from policy
            act_next = self.model.actor(
                next_obs_k, 
                self.model.vae.decode(next_obs_k)
            )
        
        feat_next = self.feature_encoder(next_obs_k, act_next)
        
        # Negative pairs: random samples from batch
        batch_size = obs_k.shape[0]
        rand_idx = torch.randperm(batch_size, device=self.device)
        feat_negative = feat_current[rand_idx]
        
        # InfoNCE loss
        # Positive similarity
        logits_pos = torch.sum(feat_current * feat_next, dim=-1) / temperature
        # Negative similarity
        logits_neg = torch.sum(feat_current * feat_negative, dim=-1) / temperature
        
        # Contrastive loss
        labels = torch.ones(batch_size, device=self.device)
        loss = F.binary_cross_entropy_with_logits(
            torch.stack([logits_pos, logits_neg], dim=1),
            torch.stack([labels, 1 - labels], dim=1)
        )
        
        self.encoder_optim.zero_grad()
        loss.backward()
        self.encoder_optim.step()
        
        return loss.item()
    
    def train_ensemble(
        self,
        batch_keep: Tuple,
        batch_forget: Tuple
    ) -> float:
        """Train ensemble critics for uncertainty estimation"""
        if not self.use_confidence_weighting:
            return 0.0
        
        obs_k, next_obs_k, act_k, rew_k, _, done_k = batch_keep
        obs_f, next_obs_f, act_f, rew_f, _, done_f = batch_forget
        
        # Combine batches
        obs = torch.cat([obs_k, obs_f], dim=0)
        next_obs = torch.cat([next_obs_k, next_obs_f], dim=0)
        act = torch.cat([act_k, act_f], dim=0)
        rew = torch.cat([rew_k, rew_f], dim=0)
        done = torch.cat([done_k, done_f], dim=0)
        
        # Standard TD learning for ensemble
        with torch.no_grad():
            next_act = self.model.actor(next_obs, self.model.vae.decode(next_obs))
            q_next = self.q_ensemble(next_obs, next_act)
            q_target = rew + self.model.gamma * (1 - done) * q_next
        
        q_pred = self.q_ensemble(obs, act)
        loss = F.mse_loss(q_pred, q_target)
        
        self.ensemble_optim.zero_grad()
        loss.backward()
        self.ensemble_optim.step()
        
        return loss.item()
    
    def compute_relabeled_targets(
        self,
        batch_forget: Tuple,
        batch_keep: Tuple,
        progress_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relabeled rewards and costs for forget data
        """
        # Get relabeled values
        if self.relabel_strategy == 'confidence' and self.use_confidence_weighting:
            rew_f_relabeled, cost_f_relabeled = self.relabeler.relabel(
                batch_forget, batch_keep,
                self.feature_encoder, self.q_ensemble
            )
        else:
            rew_f_relabeled, cost_f_relabeled = self.relabeler.relabel(
                batch_forget, batch_keep, self.feature_encoder
            )
        
        _, _, _, rew_f_original, cost_f_original, _ = batch_forget
        
        # IMPROVEMENT 1: Clip relabeling to prevent too aggressive changes
        max_change_ratio = 1.0  # Don't change more than 30% per update
        
        rew_diff = rew_f_relabeled - rew_f_original
        rew_diff_clipped = torch.clamp(rew_diff, 
                                    -max_change_ratio * torch.abs(rew_f_original),
                                    max_change_ratio * torch.abs(rew_f_original))
        rew_f_relabeled_clipped = rew_f_original + rew_diff_clipped
        
        cost_diff = cost_f_relabeled - cost_f_original
        cost_diff_clipped = torch.clamp(cost_diff,
                                        -max_change_ratio * torch.abs(cost_f_original),
                                        max_change_ratio * torch.abs(cost_f_original))
        cost_f_relabeled_clipped = cost_f_original + cost_diff_clipped
        
        # Gradual transition from clipped relabeled to full relabeled
        if self.use_gradual_relabeling:
            alpha = min(1.0, progress_ratio)
            
            # Early phase: use clipped relabeling
            # Late phase: use full relabeling
            beta = min(1.0, progress_ratio * 2.0)  # Faster ramp-up for full relabeling
            
            rew_f_final = (1 - alpha) * rew_f_original + alpha * (
                (1 - beta) * rew_f_relabeled_clipped + beta * rew_f_relabeled
            )
            cost_f_final = (1 - alpha) * cost_f_original + alpha * (
                (1 - beta) * cost_f_relabeled_clipped + beta * cost_f_relabeled
            )
        else:
            rew_f_final = rew_f_relabeled_clipped
            cost_f_final = cost_f_relabeled_clipped
        
        return rew_f_final, cost_f_final
    
    def trajdeleter_unlearn_one_step(
        self,
        batch_forget: Tuple,
        batch_keep: Tuple,
        *,
        lambda_cost: float = 1.0,
        alpha_q_keep: float = 1.0,
        alpha_q_forget: float = 1.0,  # Equal weight to relabeled data
        alpha_cq_keep: float = 1.0,
        alpha_cq_forget: float = 1.0,
        alpha_pg_keep: float = 1.0,
        alpha_pg_forget: float = 0.1,
        max_grad_norm: float = 1.0,
        use_policy_gradient: bool = True,
        verbose: bool = False,
        unlearn_method: str = "relabel",
    ) -> Dict[str, float]:
        """
        Main unlearning step with relabeling
        """
        device = self.device
        
        # Unpack batches
        obs_f, next_obs_f, act_f, rew_f, cost_f, done_f = batch_forget
        obs_k, next_obs_k, act_k, rew_k, cost_k, done_k = batch_keep
        
        # Move to device
        obs_f = obs_f.to(device)
        next_obs_f = next_obs_f.to(device)
        act_f = act_f.to(device)
        rew_f = rew_f.to(device)
        cost_f = cost_f.to(device)
        done_f = done_f.to(device)
        
        obs_k = obs_k.to(device)
        next_obs_k = next_obs_k.to(device)
        act_k = act_k.to(device)
        rew_k = rew_k.to(device)
        cost_k = cost_k.to(device)
        done_k = done_k.to(device)
        
        # Update counters
        self.current_step += 1
        self.update_counter += 1
        progress_ratio = self.current_step / self.gradual_steps
        
        # ===== FEATURE ENCODER TRAINING =====
        loss_encoder = 0.0
        if (self.use_feature_encoder and 
            self.update_counter % self.encoder_update_freq == 0):
            loss_encoder = self.train_feature_encoder(
                (obs_k, next_obs_k, act_k, rew_k, cost_k, done_k)
            )
        
        # ===== ENSEMBLE TRAINING (for confidence) =====
        loss_ensemble = 0.0
        if self.use_confidence_weighting:
            loss_ensemble = self.train_ensemble(
                (obs_k, next_obs_k, act_k, rew_k, cost_k, done_k),
                (obs_f, next_obs_f, act_f, rew_f, cost_f, done_f)
            )
        
        # ===== VAE TRAINING =====
        # Keep your original VAE training
        # recon, mean, std = self.model.vae(obs_f, act_f)
        # recon_loss = F.mse_loss(recon, act_f)
        # KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # loss_vae_f = recon_loss + self.model.beta * KL_loss
        
        # recon, mean, std = self.model.vae(obs_k, act_k)
        # recon_loss = F.mse_loss(recon, act_k)
        # KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # loss_vae_k = recon_loss + self.model.beta * KL_loss
        
        # self.model.vae_optim.zero_grad()
        # loss_vae_k.backward()
        # self.model.vae_optim.step()
        
        # ===== RELABELING =====
        rew_f_relabeled, cost_f_relabeled = self.compute_relabeled_targets(
            (obs_f, next_obs_f, act_f, rew_f, cost_f, done_f),
            (obs_k, next_obs_k, act_k, rew_k, cost_k, done_k),
            progress_ratio
        )
        
        # ===== CRITIC TRAINING WITH RELABELED DATA =====
        for p in self.model.actor.parameters():
            p.requires_grad = False
        for p in self.model.vae.parameters():
            p.requires_grad = False
        
        # Reward Critic
        _, _, q1_f, q2_f = self.model.critic.predict(obs_f, act_f)
        _, _, q1_k, q2_k = self.model.critic.predict(obs_k, act_k)
        
        with torch.no_grad():
            Bf, Bk = next_obs_f.shape[0], next_obs_k.shape[0]
            
            def target_q(next_obs):
                obs_rep = torch.repeat_interleave(
                    next_obs, self.model.sample_action_num, 0
                )
                act_next = self.model.actor_old(
                    obs_rep, self.model.vae.decode(obs_rep)
                )
                q1_t, q2_t, _, _ = self.model.critic_old.predict(obs_rep, act_next)
                q_t = (self.model.lmbda * torch.min(q1_t, q2_t) + 
                       (1. - self.model.lmbda) * torch.max(q1_t, q2_t))
                return q_t
            
            # Targets for keep (standard)
            q_t_k = target_q(next_obs_k).reshape(Bk, -1).max(1)[0]
            backup_k = rew_k + self.model.gamma * (1 - done_k) * q_t_k
            
            # Targets for forget (RELABELED rewards!)
            q_t_f = target_q(next_obs_f).reshape(Bf, -1).max(1)[0]
            backup_f = rew_f_relabeled + self.model.gamma * (1 - done_f) * q_t_f
        
        loss_q_keep = (self.model.critic.loss(backup_k, q1_k) + 
                       self.model.critic.loss(backup_k, q2_k))
        loss_q_forget = (self.model.critic.loss(backup_f, q1_f) + 
                         self.model.critic.loss(backup_f, q2_f))
        
        loss_critic_total = (alpha_q_keep * loss_q_keep + 
                            alpha_q_forget * loss_q_forget)
        
        self.model.critic_optim.zero_grad(set_to_none=True)
        loss_critic_total.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), max_grad_norm
            )
        self.model.critic_optim.step()
        
        # Cost Critic
        _, _, cq1_f, cq2_f = self.model.cost_critic.predict(obs_f, act_f)
        _, _, cq1_k, cq2_k = self.model.cost_critic.predict(obs_k, act_k)
        
        with torch.no_grad():
            def target_cq(next_obs):
                obs_rep = torch.repeat_interleave(
                    next_obs, self.model.sample_action_num, 0
                )
                act_next = self.model.actor_old(
                    obs_rep, self.model.vae.decode(obs_rep)
                )
                c1_t, c2_t, _, _ = self.model.cost_critic_old.predict(obs_rep, act_next)
                c_t = (self.model.lmbda * torch.min(c1_t, c2_t) + 
                       (1. - self.model.lmbda) * torch.max(c1_t, c2_t))
                return c_t
            
            # Targets for keep (standard)
            cq_t_k = target_cq(next_obs_k).reshape(Bk, -1).max(1)[0]
            backup_c_k = cost_k + self.model.gamma * (1 - done_k) * cq_t_k
            
            # Targets for forget (RELABELED costs!)
            cq_t_f = target_cq(next_obs_f).reshape(Bf, -1).max(1)[0]
            backup_c_f = cost_f_relabeled + self.model.gamma * (1 - done_f) * cq_t_f
        
        loss_cq_keep = (self.model.cost_critic.loss(backup_c_k, cq1_k) + 
                        self.model.cost_critic.loss(backup_c_k, cq2_k))
        loss_cq_forget = (self.model.cost_critic.loss(backup_c_f, cq1_f) + 
                          self.model.cost_critic.loss(backup_c_f, cq2_f))
        
        loss_cost_critic_total = (alpha_cq_keep * loss_cq_keep + 
                                  alpha_cq_forget * loss_cq_forget)
        
        self.model.cost_critic_optim.zero_grad(set_to_none=True)
        loss_cost_critic_total.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.cost_critic.parameters(), max_grad_norm
            )
        self.model.cost_critic_optim.step()
        
        # ===== ACTOR TRAINING =====
        for p in self.model.actor.parameters():
            p.requires_grad = True
        for p in self.model.critic.parameters():
            p.requires_grad = False
        for p in self.model.cost_critic.parameters():
            p.requires_grad = False
        
        if use_policy_gradient:
            self.model.actor.train()
            
            vae_decode_f = self.model.vae.decode(obs_f).detach()
            vae_decode_k = self.model.vae.decode(obs_k).detach()
            
            log_prob_k, mu_k = self.compute_log_prob(obs_k, act_k, vae_decode_k)
            log_prob_f, mu_f = self.compute_log_prob(obs_f, act_f, vae_decode_f)
            
            A_r_k, A_c_k = self.safe_advantage(obs_k, act_k, K=None)
            A_r_f, A_c_f = self.safe_advantage(obs_f, act_f, K=None)
            
            # IMPROVEMENT 3: Adaptive cost penalty
            # Initialize adaptive lambda if not exists
            if not hasattr(self, '_adaptive_lambda_cost'):
                self._adaptive_lambda_cost = lambda_cost
                self._target_cost = 10.0  # Your cost limit
            
            # Update adaptive lambda based on recent cost performance
            # If cost is high, increase penalty; if low, decrease
            current_cost_estimate = A_c_k.mean().item()
            if current_cost_estimate > self._target_cost:
                self._adaptive_lambda_cost = min(10.0, self._adaptive_lambda_cost * 1.05)
            else:
                self._adaptive_lambda_cost = max(0.5, self._adaptive_lambda_cost * 0.98)
            
            # Standard policy gradient with adaptive cost penalty
            pg_loss_keep = (-(log_prob_k * A_r_k).mean() + 
                        self._adaptive_lambda_cost * (log_prob_k * A_c_k).mean())
            
            pg_loss_forget = (-(log_prob_f * A_r_f).mean() + 
                            self._adaptive_lambda_cost * (log_prob_f * A_c_f).mean())
            
            loss_actor = (alpha_pg_keep * pg_loss_keep + 
                        alpha_pg_forget * pg_loss_forget)
            
            # Store adaptive lambda for logging
            adaptive_lambda_for_log = self._adaptive_lambda_cost
                    
            # Compute gaps for monitoring
            Ar_gap = (A_r_k.mean() - A_r_f.mean())
            Ac_gap = (A_c_k.mean() - A_c_f.mean())
            
            if verbose:
                print(f'Progress: {progress_ratio:.2%} | '
                      f'Ar_gap: {Ar_gap.item():.3f} | '
                      f'Ac_gap: {Ac_gap.item():.3f}')
        
        self.model.actor_optim.zero_grad(set_to_none=True)
        loss_actor.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.actor.parameters(), max_grad_norm
            )
        self.model.actor_optim.step()
        
        # Re-enable all gradients
        for p in self.model.critic.parameters():
            p.requires_grad = True
        for p in self.model.cost_critic.parameters():
            p.requires_grad = True
        for p in self.model.vae.parameters():
            p.requires_grad = True
        
        # ===== MONITORING =====
        # Compute average relabeling distance for monitoring
        avg_relabel_distance = 0.0
        if self.feature_encoder is not None:
            with torch.no_grad():
                feat_f = self.feature_encoder(obs_f, act_f)
                feat_k = self.feature_encoder(obs_k, act_k)
                distances = torch.cdist(feat_f, feat_k)
                avg_relabel_distance = distances.min(dim=1)[0].mean().item()
        
        return {
            'Ac_gap': float(Ac_gap.detach().cpu()),
            'Ar_gap': float(Ar_gap.detach().cpu()),
            'loss_actor': float(loss_actor.detach().cpu()),
            'loss_q_keep': float(loss_q_keep.detach().cpu()),
            'loss_q_forget': float(loss_q_forget.detach().cpu()),
            'loss_cq_keep': float(loss_cq_keep.detach().cpu()),
            'loss_cq_forget': float(loss_cq_forget.detach().cpu()),
            'loss_encoder': float(loss_encoder),
            'loss_ensemble': float(loss_ensemble),
            'avg_relabel_distance': float(avg_relabel_distance),
            'progress': float(progress_ratio),
            'rew_f_original': float(rew_f.mean().detach().cpu()),
            'rew_f_relabeled': float(rew_f_relabeled.mean().detach().cpu()),
            'cost_f_original': float(cost_f.mean().detach().cpu()),
            'cost_f_relabeled': float(cost_f_relabeled.mean().detach().cpu()),
        }

    def compute_log_prob(self, obs, actions, vae_decode):
        """Compute log probability of actions under the policy"""
        model_self = self.model
        mean = model_self.actor(obs, vae_decode)
        
        # Compute Gaussian log probability
        # Assuming the actor outputs mean and has a fixed std or learned std
        # This is a simplified version - adjust based on your actor architecture
        
        # If actor has a method to get both mean and std:
        if hasattr(model_self.actor, 'get_dist'):
            dist = model_self.actor.get_dist(obs, vae_decode)
            log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            return log_prob, mean
        else:
            # Fixed std approach (you may need to adjust this)
            std = torch.ones_like(mean) * 0.1  # or use a learned parameter
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            return log_prob, mean
        
    def safe_advantage(self, obs, actions, K=10):
        model_self = self.model
        with torch.no_grad():
            q1, q2, _, _ = model_self.critic.predict(obs, actions)
            c1, c2, _, _ = model_self.cost_critic.predict(obs, actions)
            q = torch.min(q1, q2)
            c = torch.min(c1, c2)
            
            if K is None or K <= 0:
                K = 10
            
            vae_decode = model_self.vae.decode(obs)
            obs_rep = obs.repeat_interleave(K, dim=0)
            vae_rep = vae_decode.repeat_interleave(K, dim=0)
            actions_sampled = model_self.actor(obs_rep, vae_rep)
            
            q1_s, q2_s, _, _ = model_self.critic.predict(obs_rep, actions_sampled)
            c1_s, c2_s, _, _ = model_self.cost_critic.predict(obs_rep, actions_sampled)
            
            q_sampled = torch.min(q1_s, q2_s).reshape(-1, K)
            c_sampled = torch.min(c1_s, c2_s).reshape(-1, K)
            
            v_q = q_sampled.max(dim=1, keepdim=True)[0]
            v_c = c_sampled.max(dim=1, keepdim=True)[0]
            
            adv_q = q - v_q
            adv_c = c - v_c
            
            # IMPROVEMENT 2: Normalize advantages
            def normalize_adv(adv, epsilon=1e-8):
                return (adv - adv.mean()) / (adv.std() + epsilon)
            
            adv_q = normalize_adv(adv_q)
            adv_c = normalize_adv(adv_c)
        
        return adv_q.detach(), adv_c.detach()
    
    