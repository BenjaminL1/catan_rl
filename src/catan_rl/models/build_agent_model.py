"""
Factory function: one call to create a fully initialized CatanPolicy.
"""

from catan_rl.models.policy import CatanPolicy

DEFAULT_MODEL_CONFIG = {
    "obs_output_dim": 512,
    "tile_in_dim": 79,
    "tile_model_dim": 128,
    "curr_player_main_in_dim": 166,
    "other_player_main_in_dim": 173,
    "dev_card_embed_dim": 64,
    "dev_card_model_dim": 64,
    "tile_model_num_heads": 4,
    "proj_dev_card_dim": 25,
    "dev_card_model_num_heads": 4,
    "tile_encoder_num_layers": 2,
    "proj_tile_dim": 25,
    "action_head_hidden_dim": 128,
    "value_hidden_dims": (256, 128),
    "dropout": 0.0,
    # Phase 1.4: dev-card encoding mode. True = legacy MHA pipeline (matches
    # checkpoint_07390040.pt). False = count-based MLP encoder (smaller, faster,
    # permutation-invariant by construction). Default True to preserve compat.
    "use_devcard_mha": True,
    "max_dev_seq": 16,
    "dev_card_vocab_excl_pad": 5,
}


def build_agent_model(device: str = "cpu", **overrides) -> CatanPolicy:
    """Build a CatanPolicy with sensible defaults, move to device, and return.

    Args:
        device:     "cpu", "cuda", or "mps".
        **overrides: any key from DEFAULT_MODEL_CONFIG to change.
                     Example: build_agent_model(obs_output_dim=256)

    Returns:
        Fully initialized CatanPolicy on the specified device.
    """
    cfg = {**DEFAULT_MODEL_CONFIG, **overrides}

    policy = CatanPolicy(
        obs_output_dim=cfg["obs_output_dim"],
        value_hidden_dims=cfg["value_hidden_dims"],
        action_head_hidden_dim=cfg["action_head_hidden_dim"],
        tile_in_dim=cfg["tile_in_dim"],
        tile_model_dim=cfg["tile_model_dim"],
        curr_player_main_in_dim=cfg["curr_player_main_in_dim"],
        other_player_main_in_dim=cfg["other_player_main_in_dim"],
        dev_card_embed_dim=cfg["dev_card_embed_dim"],
        dev_card_model_dim=cfg["dev_card_model_dim"],
        tile_model_num_heads=cfg["tile_model_num_heads"],
        proj_dev_card_dim=cfg["proj_dev_card_dim"],
        dev_card_model_num_heads=cfg["dev_card_model_num_heads"],
        tile_encoder_num_layers=cfg["tile_encoder_num_layers"],
        proj_tile_dim=cfg["proj_tile_dim"],
        dropout=cfg["dropout"],
        use_devcard_mha=cfg["use_devcard_mha"],
        max_dev_seq=cfg["max_dev_seq"],
        dev_card_vocab_excl_pad=cfg["dev_card_vocab_excl_pad"],
    )

    # Count parameters for logging
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"CatanPolicy created: {n_params:,} trainable parameters")

    return policy.to(device)
