from .edcoder import PreModel


def build_model(args):
    model = PreModel(
        in_dim=args.num_features,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        nhead=args.num_heads,
        nhead_out=args.num_out_heads,
        activation=args.activation,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        negative_slope=args.negative_slope,
        residual=args.residual,
        encoder_type=args.encoder,
        decoder_type=args.decoder,
        mask_rate_node=args.mask_rate_node,
        mask_rate_edge=args.mask_rate_edge,
        norm=args.norm,
        loss_fn=args.loss_fn,
        drop_edge_rate=args.drop_edge_rate,
        replace_rate=args.replace_rate,
        alpha_l=args.alpha_l,
        concat_hidden=args.concat_hidden,
        lambda_=args.lambda_,
    )
    return model
