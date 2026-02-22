import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MSTCN(nn.Module):
    """
    Multi-Scale Temporal Convolution Network.

    __init__ shape contract:
    - in_channels: x [B, N, W] is treated as per-node 1D signal
    - branch_channels: each branch conv output channel C

    forward shape contract:
    - Input:  x [B, N, W]
    - Output: node_feat [B, N, D], where D = 4 * C
    """

    def __init__(self, branch_channels: int = 16):
        super().__init__()
        self.branch_channels = branch_channels
        self.kernel_sizes = [2, 3, 5, 7]

        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_channels=1, out_channels=branch_channels, kernel_size=k, padding=0)
                for k in self.kernel_sizes
            ]
        )

    @staticmethod
    def _manual_same_pad(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        x: [B*N, 1, W] -> padded so that Conv1d(kernel_size=k, padding=0) keeps length W.
        """
        total_pad = kernel_size - 1
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        return F.pad(x, (pad_left, pad_right))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_nodes, win = x.shape

        x_reshaped = x.reshape(bsz * num_nodes, 1, win)
        branch_feats = []

        for conv, k in zip(self.branches, self.kernel_sizes):
            x_pad = self._manual_same_pad(x_reshaped, k)
            out = conv(x_pad)
            if out.size(-1) != win:
                raise RuntimeError(f"MSTCN branch kernel={k} produced length {out.size(-1)} != {win}")

            out = out.reshape(bsz, num_nodes, self.branch_channels, win)
            out = out.mean(dim=-1)
            branch_feats.append(out)

        node_feat = torch.cat(branch_feats, dim=-1)
        return node_feat


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with KL sparsity penalty.

    __init__ shape contract:
    - Input node feature dim D
    - Latent dim Z
    - Reconstruction output dim W (window size)

    forward shape contract:
    - Input:  node_feat [B, N, D]
    - Output: z [B, N, Z], reconstructed_vals [B, N, W], kl_sparsity scalar, sparsity_dev [B, N]
    """

    def __init__(self, in_dim: int, latent_dim: int, window_size: int, rho: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.encoder = nn.Linear(in_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, window_size)
        self.rho = float(rho)
        self.eps = float(eps)

    def kl_divergence(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, N, Z] -> KL(rho || rho_hat) scalar
        rho_hat_l = mean_{B,N}(sigmoid(z)) for each latent l
        """
        rho_hat = torch.sigmoid(z).mean(dim=(0, 1))
        rho_hat = torch.clamp(rho_hat, self.eps, 1.0 - self.eps)

        rho = torch.full_like(rho_hat, self.rho)
        kl = rho * torch.log(rho / rho_hat) + (1.0 - rho) * torch.log((1.0 - rho) / (1.0 - rho_hat))
        return kl.sum()

    def forward(self, node_feat: torch.Tensor):
        z = self.encoder(node_feat)
        reconstructed_vals = self.decoder(z)
        kl_sparsity = self.kl_divergence(z)
        activation = torch.sigmoid(z)
        rho_target = torch.full_like(activation, self.rho)
        sparsity_dev = torch.mean(torch.abs(activation - rho_target), dim=-1)
        return z, reconstructed_vals, kl_sparsity, sparsity_dev


class LinearProjection(nn.Module):
    """
    Linear projection layer for use_sae=0 ablation.
    Projects MSTCN output [B, N, D] to [B, N, Z] for GCN/GAT input.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, node_feat: torch.Tensor):
        """
        Input:  node_feat [B, N, D]
        Output: z [B, N, Z]
        """
        z = self.proj(node_feat)
        return z


class GraphLearning(nn.Module):
    """
    Graph learning from trainable sensor embeddings.

    __init__ shape contract:
    - Learnable sensor_embeddings [N, E]

    forward shape contract:
    - Output: A [N, N], sensor_embeddings [N, E]
    """

    def __init__(self, num_nodes: int, embed_dim: int, topk: int):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.embed_dim = int(embed_dim)
        self.topk = int(topk)

        if not (1 <= self.topk < self.num_nodes):
            raise ValueError(f"topk must satisfy 1 <= topk < N, got topk={self.topk}, N={self.num_nodes}")

        self.sensor_embeddings = nn.Parameter(torch.empty(self.num_nodes, self.embed_dim))
        nn.init.xavier_uniform_(self.sensor_embeddings, gain=1.0)

    def forward(self):
        emb = F.normalize(self.sensor_embeddings, p=2, dim=-1)
        sim = emb @ emb.transpose(0, 1)
        sim = sim / math.sqrt(self.embed_dim)

        # Exclude self before top-k neighbor selection, then add self-loop explicitly later.
        sim_no_self = sim.clone()
        sim_no_self.fill_diagonal_(float("-inf"))
        topk_idx = torch.topk(sim_no_self, k=self.topk, dim=-1).indices
        mask = torch.zeros_like(sim)
        mask.scatter_(1, topk_idx, 1.0)

        a = F.relu(sim) * mask
        eye = torch.eye(self.num_nodes, device=a.device, dtype=a.dtype)
        a = a + eye

        return a, self.sensor_embeddings


class DenseGATLayer(nn.Module):
    """
    Dense single-head GAT layer using adjacency mask.

    forward shape contract:
    - Input:  x [B, N, F_in], A [N, N]
    - Output: out [B, N, F_out]
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.w = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = x @ self.w

        e_src = h @ self.a_src
        e_dst = h @ self.a_dst
        e = self.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1))

        attn_mask = (a > 0).unsqueeze(0)
        e = e.masked_fill(~attn_mask, float("-inf"))

        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)

        out = alpha @ h
        return out


class DenseGAT(nn.Module):
    """
    Dense multi-head GAT (pure PyTorch).

    forward shape contract:
    - Input:  x [B, N, F_in], A [N, N]
    - Output: fused [B, N, H]
    """

    def __init__(self, in_dim: int, head_dim: int, out_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [DenseGATLayer(in_dim=in_dim, out_dim=head_dim, dropout=dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_dim * num_heads, out_dim)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        multi_head = [head(x, a) for head in self.heads]
        h = torch.cat(multi_head, dim=-1)
        fused = self.act(self.proj(h))
        return fused


class JointLoss(nn.Module):
    """
    Joint loss for forecasting + reconstruction + sparsity.

    forward shape contract:
    - predicted_vals [B, N]
    - forecast_target [B, N]
    - reconstructed_vals [B, N, W] (or None if use_sae=0)
    - recon_target [B, N, W]
    - kl_sparsity scalar (or 0 if use_sae=0)

    Returns:
    - dict with keys: total, forecast, reconstruction, sparsity
    """

    def __init__(self, lambda_forecast: float = 0.7, beta: float = 1e-3, use_sae: int = 1):
        super().__init__()
        self.lambda_forecast = float(lambda_forecast)
        self.beta = float(beta)
        self.use_sae = int(use_sae)

    def forward(
        self,
        predicted_vals: torch.Tensor,
        forecast_target: torch.Tensor,
        reconstructed_vals: torch.Tensor = None,
        recon_target: torch.Tensor = None,
        kl_sparsity: torch.Tensor = None,
    ):
        forecast_loss = F.mse_loss(predicted_vals, forecast_target)
        
        if self.use_sae:
            # Normal case: compute all losses
            recon_loss = F.mse_loss(reconstructed_vals, recon_target)
            sparsity = kl_sparsity if kl_sparsity is not None else torch.tensor(0.0, device=predicted_vals.device)
            total = self.lambda_forecast * forecast_loss + (1.0 - self.lambda_forecast) * recon_loss + self.beta * sparsity
        else:
            # Ablation case: only forecast loss
            recon_loss = torch.tensor(0.0, device=predicted_vals.device)
            sparsity = torch.tensor(0.0, device=predicted_vals.device)
            total = forecast_loss
        
        return {
            "total": total,
            "forecast": forecast_loss,
            "reconstruction": recon_loss,
            "sparsity": sparsity,
        }


class TopoFuSAGNet(nn.Module):
    """
    Topology-Fusion Sparse-AE GAT Net for MTSAD.

    Data flow (strict order):
    [B,N,W]
      -> MSTCN -> node_feat [B,N,D]
    -> SAE (or LinearProjection if use_sae=0) -> z [B,N,Z], reconstructed_vals [B,N,W] (or None), kl (or 0), sparsity_dev [B,N]
      -> GraphLearning -> A [N,N], sensor_embeddings [N,E]
      -> concat(z, sensor_embeddings_broadcast) -> gat_in [B,N,Z+E]
      -> DenseGAT -> fused [B,N,H]
      -> Forecast Head -> predicted_vals [B,N]

    forward returns:
    - predicted_vals [B,N]
    - reconstructed_vals [B,N,W] (or None if use_sae=0)
    - kl_sparsity scalar
    - sparsity_dev [B,N]
    """

    def __init__(
        self,
        num_nodes: int,
        window_size: int,
        c: int = 16,
        z_dim: int = 8,
        emb_dim: int = 8,
        gat_out_dim: int = 32,
        topk: int = 5,
        rho: float = 0.05,
        dropout: float = 0.1,
        gat_heads: int = 2,
        use_sae: int = 1,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.window_size = int(window_size)
        self.use_sae = int(use_sae)

        self.mstcn = MSTCN(branch_channels=c)
        
        if self.use_sae:
            self.sae = SparseAutoencoder(in_dim=4 * c, latent_dim=z_dim, window_size=window_size, rho=rho)
        else:
            self.proj = LinearProjection(in_dim=4 * c, out_dim=z_dim)
        
        self.graph_learning = GraphLearning(num_nodes=num_nodes, embed_dim=emb_dim, topk=topk)

        self.gat = DenseGAT(
            in_dim=z_dim + emb_dim,
            head_dim=max(8, gat_out_dim // max(1, gat_heads)),
            out_dim=gat_out_dim,
            num_heads=gat_heads,
            dropout=dropout,
        )
        self.forecast_head = nn.Linear(gat_out_dim, 1)

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B,N,W], got {tuple(x.shape)}")

        bsz, num_nodes, win = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Input N={num_nodes} does not match model num_nodes={self.num_nodes}")
        if win != self.window_size:
            raise ValueError(f"Input W={win} does not match model window_size={self.window_size}")

        node_feat = self.mstcn(x)
        
        if self.use_sae:
            z, reconstructed_vals, kl_sparsity, sparsity_dev = self.sae(node_feat)
        else:
            z = self.proj(node_feat)
            reconstructed_vals = None
            kl_sparsity = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            sparsity_dev = torch.zeros((bsz, num_nodes), device=x.device, dtype=x.dtype)

        a, sensor_embeddings = self.graph_learning()
        e_batch = sensor_embeddings.unsqueeze(0).expand(bsz, -1, -1)
        gat_in = torch.cat([z, e_batch], dim=-1)

        fused = self.gat(gat_in, a)
        predicted_vals = self.forecast_head(fused).squeeze(-1)

        return predicted_vals, reconstructed_vals, kl_sparsity, sparsity_dev


if __name__ == "__main__":
    torch.manual_seed(42)

    bsz, num_nodes, win = 8, 20, 15
    x = torch.randn(bsz, num_nodes, win)
    forecast_target = torch.randn(bsz, num_nodes)

    model = TopoFuSAGNet(
        num_nodes=num_nodes,
        window_size=win,
        c=16,
        z_dim=12,
        emb_dim=10,
        gat_out_dim=32,
        topk=5,
        rho=0.05,
        dropout=0.1,
        gat_heads=2,
    )

    predicted_vals, reconstructed_vals, kl_sparsity, sparsity_dev = model(x)

    assert predicted_vals.shape == (bsz, num_nodes)
    assert reconstructed_vals.shape == (bsz, num_nodes, win)
    assert sparsity_dev.shape == (bsz, num_nodes)

    criterion = JointLoss(lambda_forecast=0.7, beta=1e-3)
    loss_dict = criterion(
        predicted_vals=predicted_vals,
        forecast_target=forecast_target,
        reconstructed_vals=reconstructed_vals,
        recon_target=x,
        kl_sparsity=kl_sparsity,
    )

    print("predicted_vals:", predicted_vals.shape)
    print("reconstructed_vals:", reconstructed_vals.shape)
    print("kl_sparsity:", float(kl_sparsity.detach().cpu()))
    print("sparsity_dev:", sparsity_dev.shape)
    print({k: float(v.detach().cpu()) for k, v in loss_dict.items()})
