from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from utils import create_norm, drop_edge


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate_node: float = 0.3,
            mask_rate_edge: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            lambda_: float = 0.01
         ):
        super(PreModel, self).__init__()
        self._mask_rate_node = mask_rate_node
        self._mask_rate_edge = mask_rate_edge

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.lambda_ = lambda_

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, x, num_nodes, num_mask_nodes, mask_nodes):

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, mask_nodes

    def forward(self, g, x):
        num_nodes = g.num_nodes()
        perm_node = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(self._mask_rate_node * num_nodes)

        # random masking
        mask_nodes = perm_node[: num_mask_nodes]
        # keep_nodes = perm_node[num_mask_nodes: ]

        use_x, mask_nodes = self.encoding_mask_noise(x, num_nodes, num_mask_nodes, mask_nodes)

        g1=g.clone()
        g1.remove_nodes(nids=mask_nodes)

        num_edges = g.num_nodes()
        perm_edge = torch.randperm(num_edges, device=x.device)
        num_mask_edges = int(self._mask_rate_edge * num_edges)

        # random masking
        mask_edges = perm_edge[: num_mask_edges]
        keep_edges = perm_edge[num_mask_edges: ]

        g2=g.clone()
        g2.remove_edges(eids=mask_edges)

        z3, loss_GAE = self.mask_attr_prediction(x, g, use_x, mask_nodes)
        loss_CL = self.contrastive_loss(g1, g2, z3)
        loss = loss_GAE + self.lambda_ * loss_CL
        loss_item = {"loss": loss.item(), "loss_GAE": loss_GAE.item(), "loss_CL": loss_CL.item()}
        return loss, loss_GAE, loss_CL,  loss_item

    def mask_attr_prediction(self, x, pre_use_g, use_x, mask_nodes):

        encoded_representation, all_hidden = self.encoder(pre_use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            encoded_representation = torch.cat(all_hidden, dim=1)

        representation = self.encoder_to_decoder(encoded_representation)

        if self._decoder_type not in ("mlp", "linear"):
            representation[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(representation)
        else:
            recon = self.decoder(pre_use_g, representation)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return encoded_representation, loss

    def contrastive_loss(self, g1, g2, z3):
        tau1 = calculate_current_tau(self._tau1, self._tau1_decay_rate, self._tau1_decay_step)
        tau2 = calculate_current_tau(self._tau2, self._tau2_decay_rate, self._tau1_decay_step)

        z1, all_hidden_1 = self.encoder(g1, g1.ndata["feat"], return_hidden=True)
        z2, all_hidden_2 = self.encoder(g2, g2.ndata["feat"], return_hidden=True)
        if self._concat_hidden:
            z1 = torch.cat(all_hidden_1, dim=1)
            z2 = torch.cat(all_hidden_2, dim=1)

        batch_size, _ = z1.size()
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        z3_abs = z3.norm(dim=1)

        sim_matrix_intra = torch.einsum('ik,jk->ij', z1, z2) / torch.einsum('i,j->ij', z1_abs, z2_abs).clamp_(1e-6)
        sim_matrix_intra = torch.exp(sim_matrix_intra / tau1)
        sim_matrix_inter = torch.einsum('ik,jk->ij', z1, z3) / torch.einsum('i,j->ij', z1_abs, z3_abs).clamp_(1e-6)
        sim_matrix_inter = torch.exp(sim_matrix_inter / tau2)
        pos_sim = sim_matrix_intra[range(batch_size), range(batch_size)]
        # loss = pos_sim / (sim_matrix_intra.sum(dim=1) - pos_sim)
        loss = pos_sim / (sim_matrix_intra.sum(dim=1) + sim_matrix_inter.sum(dim=1))
        loss = -torch.log(loss).mean()
        return loss

    def embed(self, g, x):
        representation = self.encoder(g, x)
        return representation

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
