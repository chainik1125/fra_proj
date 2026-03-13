import json
import sys
import types


def _install_stub_module(name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    sys.modules[name] = module


_install_stub_module("fra.fra_func", attention_pattern_QK=lambda **_: None)
_install_stub_module(
    "sleepers.analysis.ft_analysis_util",
    DEFAULT_HOOK_POINTS=[
        "blocks.0.hook_resid_pre",
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_post",
        "blocks.2.hook_resid_post",
    ],
    get_activations=lambda *args, **kwargs: None,
    load_wandb_crosscoder=lambda *args, **kwargs: (None, None),
)
_install_stub_module("sleepers.scripts.llms", build_llm_lora=lambda *args, **kwargs: None)

from fra_sleeper.ablation import (
    AblationPair,
    _apply_attention_head_zeroing,
    _build_ablation_config,
    _compute_pair_interaction_delta,
    _load_layer_ablation_plans,
    _resolve_zero_attention_heads,
)
from fra_sleeper.analysis import _build_config_from_mapping, iter_layer_configs, layer_output_dir, summary_path_for_variant


def test_build_config_parses_named_multi_layers() -> None:
    config = _build_config_from_mapping(
        {
            "output_dir": "artifacts/test_multi",
            "layers": [
                {"name": "early_context", "layer": 0},
                {"name": "late_context", "layer": 2},
            ],
            "key_features": [628, 832],
            "device": "cpu",
        }
    )

    assert config.use_layer_subdirs is True
    assert [(layer.name, layer.layer) for layer in config.layers] == [
        ("early_context", 0),
        ("late_context", 2),
    ]

    layer_configs = iter_layer_configs(config)
    assert [layer_config.layer for layer_config in layer_configs] == [0, 2]
    assert [str(layer_output_dir(layer_config)) for layer_config in layer_configs] == [
        "artifacts/test_multi/early_context",
        "artifacts/test_multi/late_context",
    ]


def test_load_layer_ablation_plans_reads_layer_specific_summaries(tmp_path) -> None:
    config = _build_config_from_mapping(
        {
            "output_dir": str(tmp_path),
            "layers": [
                {"name": "resid_pre", "layer": 0},
                {"name": "resid_post", "layer": 1},
            ],
            "key_features": [628],
            "device": "cpu",
        }
    )
    ablation_config = _build_ablation_config(
        {
            "ablation": {
                "enabled": True,
                "top_pairs_per_key": 2,
            }
        }
    )

    layer_summaries = {
        "resid_pre": [2793, 647],
        "resid_post": [1880, 447],
    }
    for layer_config in iter_layer_configs(config):
        output_dir = layer_output_dir(layer_config)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_path_for_variant(output_dir, "sleeper_model_plus_sleeper_data")
        summary_path.write_text(
            json.dumps(
                {
                    "top_pairs_by_key_feature": {
                        "628": [
                            {
                                "key_feature": 628,
                                "paired_feature": feature_id,
                                "abs_interaction_score": float(index + 1),
                            }
                            for index, feature_id in enumerate(layer_summaries[layer_config.layer_name])
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

    plans = _load_layer_ablation_plans(config, ablation_config)

    assert [(plan.layer_name, plan.layer) for plan in plans] == [("resid_pre", 0), ("resid_post", 1)]
    assert [[pair.paired_feature for pair in plan.pairs] for plan in plans] == [
        [2793, 647],
        [1880, 447],
    ]


def test_build_ablation_config_parses_zero_attention_heads() -> None:
    config = _build_ablation_config(
        {
            "ablation": {
                "enabled": True,
                "method": "head_output_zeroing",
                "zero_attention_heads": [0, 3, 7],
            }
        }
    )

    assert config.method == "head_output_zeroing"
    assert config.zero_attention_heads == [0, 3, 7]
    assert config.zero_all_attention_heads is False

    all_config = _build_ablation_config(
        {
            "ablation": {
                "enabled": True,
                "method": "head_output_zeroing",
                "zero_attention_heads": "all",
            }
        }
    )

    assert all_config.zero_attention_heads == []
    assert all_config.zero_all_attention_heads is True


def test_resolve_zero_attention_heads_expands_all_and_validates_range() -> None:
    assert _resolve_zero_attention_heads([], zero_all_attention_heads=True, total_heads=4) == [0, 1, 2, 3]

    try:
        _resolve_zero_attention_heads([0, 4], zero_all_attention_heads=False, total_heads=4)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("Expected out-of-range head selection to fail")


def test_apply_attention_head_zeroing_sets_selected_heads_to_zero() -> None:
    import torch

    head_output = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)

    updated = _apply_attention_head_zeroing(head_output, [1, 3])

    assert torch.equal(updated[:, :, 0, :], head_output[:, :, 0, :])
    assert torch.equal(updated[:, :, 2, :], head_output[:, :, 2, :])
    assert torch.count_nonzero(updated[:, :, 1, :]) == 0
    assert torch.count_nonzero(updated[:, :, 3, :]) == 0


def test_compute_pair_interaction_delta_only_when_pair_is_active() -> None:
    import torch

    hidden = torch.tensor(
        [
            [2.0, 3.0, 0.0],
            [0.0, 5.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    coeff = torch.zeros((3, 3))
    coeff[0, 1] = 10.0
    coeff[1, 0] = 7.0

    delta, stats = _compute_pair_interaction_delta(
        hidden_bh=hidden,
        coeff_matrix=coeff,
        pairs=[AblationPair(key_feature=0, paired_feature=1, score=1.0)],
        key_activation_threshold=0.0,
        paired_activation_threshold=0.0,
        suppression_factor=1.0,
    )

    expected = torch.tensor(
        [
            [102.0, 100.0, 84.0],
            [70.0, 0.0, 140.0],
            [120.0, 200.0, 0.0],
        ]
    )
    assert torch.equal(delta, expected)
    assert stats["paired_edited_positions"] == 8
