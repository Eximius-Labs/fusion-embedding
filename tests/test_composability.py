"""Co-loaded multi-pack composability (docs/sensor_extension_plan.md §6).

The exclusivity invariant, verified at the unit level before any GPU run: with an
AUDIO pack and a THERMAL pack co-loaded on the same decoder layers (both with
nonzero weights),

1. an audio-scoped forward is BITWISE identical to the same forward on a model
   carrying the audio pack alone;
2. a thermal-scoped forward is BITWISE identical to the thermal-pack-alone model;
3. an unscoped forward (all gates closed) is BITWISE identical to the raw base;
4. the same holds when the audio pack is attached through the LEGACY single-gate
   path (``attach_gated_adapters``, the released FE2 configuration) and only the
   thermal pack goes through the ``AdapterPacks`` registry — the deployment mix;
5. training-style gradients under co-load reach only the scoped pack.

Mixed inputs whose tokens would activate two gates at once are OUTSIDE the
guarantee and deliberately untested (see research_composability_prior_art.md §6).
"""

import copy

import torch
import torch.nn as nn

from fusion_embedding.adapters import AdapterPacks, attach_gated_adapters
from fusion_embedding._tiny import build_tiny_components
from fusion_embedding.config import FusionConfig

RANK = 8


def _tiny_base():
    cfg = FusionConfig.tiny()
    embed_tokens, base_lm, _ = build_tiny_components(cfg)
    for p in base_lm.parameters():
        p.requires_grad_(False)
    return cfg, embed_tokens, base_lm


def _embeds(cfg, embed_tokens, B=3, S=7, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 60, (B, S), generator=g)
    ids[:, -1] = cfg.eos_id
    return embed_tokens(ids), torch.ones(B, S, dtype=torch.long)


def _fill_up(adapters: nn.ModuleList, seed: int):
    """Give a pack nonzero 'up' weights deterministically (zero-init otherwise)."""
    g = torch.Generator().manual_seed(seed)
    for m in adapters.modules():
        if isinstance(m, nn.Linear) and torch.count_nonzero(m.weight) == 0:
            m.weight.data = torch.randn(m.weight.shape, generator=g) * 0.02


def _pack_state(packs: AdapterPacks, name: str):
    return copy.deepcopy(getattr(packs, f"{name}_adapters").state_dict())


def test_coloaded_matches_each_single_pack_and_base():
    cfg, embed_tokens, base_lm = _tiny_base()
    x, mask = _embeds(cfg, embed_tokens)
    base_ref = base_lm(inputs_embeds=x, attention_mask=mask)

    # single-pack references, weights fixed by seed
    solo_a = AdapterPacks()
    a_ad, _ = solo_a.add_pack("audio", base_lm, cfg.d_llm, RANK)
    _fill_up(a_ad, seed=11)
    with solo_a.scope("audio"):
        audio_ref = base_lm(inputs_embeds=x, attention_mask=mask)
    a_sd = _pack_state(solo_a, "audio")
    for h in solo_a._handles:
        h.remove()

    solo_t = AdapterPacks()
    t_ad, _ = solo_t.add_pack("thermal", base_lm, cfg.d_llm, RANK)
    _fill_up(t_ad, seed=22)
    with solo_t.scope("thermal"):
        thermal_ref = base_lm(inputs_embeds=x, attention_mask=mask)
    t_sd = _pack_state(solo_t, "thermal")
    for h in solo_t._handles:
        h.remove()

    # co-loaded: same weights, both packs on the same layers
    both = AdapterPacks()
    ba, _ = both.add_pack("audio", base_lm, cfg.d_llm, RANK)
    bt, _ = both.add_pack("thermal", base_lm, cfg.d_llm, RANK)
    ba.load_state_dict(a_sd)
    bt.load_state_dict(t_sd)

    with both.scope("audio"):
        audio_co = base_lm(inputs_embeds=x, attention_mask=mask)
    with both.scope("thermal"):
        thermal_co = base_lm(inputs_embeds=x, attention_mask=mask)
    closed_co = base_lm(inputs_embeds=x, attention_mask=mask)

    assert torch.equal(audio_co, audio_ref), "audio-scoped co-loaded != audio-only model"
    assert torch.equal(thermal_co, thermal_ref), "thermal-scoped co-loaded != thermal-only model"
    assert torch.equal(closed_co, base_ref), "all-gates-closed co-loaded != raw base"
    assert not torch.equal(audio_ref, thermal_ref), "packs must actually differ"


def test_deployment_mix_legacy_audio_plus_registry_thermal():
    """The released FE2 audio pack uses the legacy single-gate attach; thermal arrives
    via the registry. The invariant must hold for exactly this mix."""
    cfg, embed_tokens, base_lm = _tiny_base()
    x, mask = _embeds(cfg, embed_tokens, seed=4)
    base_ref = base_lm(inputs_embeds=x, attention_mask=mask)

    audio_ad, audio_gate, audio_handles = attach_gated_adapters(base_lm, cfg.d_llm, RANK)
    _fill_up(audio_ad, seed=31)
    with audio_gate:
        audio_ref = base_lm(inputs_embeds=x, attention_mask=mask)

    packs = AdapterPacks()
    t_ad, _ = packs.add_pack("thermal", base_lm, cfg.d_llm, RANK)
    _fill_up(t_ad, seed=32)

    with audio_gate:                                   # legacy gate, thermal closed
        audio_co = base_lm(inputs_embeds=x, attention_mask=mask)
    assert torch.equal(audio_co, audio_ref), "legacy-audio forward changed after thermal attach"

    # thermal-only reference needs the audio hooks gone; then compare against the
    # co-loaded thermal forward taken BEFORE removal (order: co-loaded first).
    with packs.scope("thermal"):
        thermal_co = base_lm(inputs_embeds=x, attention_mask=mask)
    closed_co = base_lm(inputs_embeds=x, attention_mask=mask)
    for h in audio_handles:
        h.remove()
    with packs.scope("thermal"):
        thermal_solo = base_lm(inputs_embeds=x, attention_mask=mask)

    assert torch.equal(thermal_co, thermal_solo), "registry-thermal forward affected by legacy audio hooks"
    assert torch.equal(closed_co, base_ref), "closed-gate mix != raw base"


def test_gradient_isolation_under_coload():
    """Training one pack while the other is co-loaded must leave the other untouched."""
    cfg, embed_tokens, base_lm = _tiny_base()
    x, mask = _embeds(cfg, embed_tokens, seed=6)
    packs = AdapterPacks()
    a_ad, _ = packs.add_pack("audio", base_lm, cfg.d_llm, RANK)
    t_ad, _ = packs.add_pack("thermal", base_lm, cfg.d_llm, RANK)
    _fill_up(a_ad, seed=41)
    _fill_up(t_ad, seed=42)

    with packs.scope("thermal"):
        out = base_lm(inputs_embeds=x, attention_mask=mask)
    out.pow(2).mean().backward()

    assert all(p.grad is None for p in packs.parameters_of("audio")), \
        "co-loaded audio pack received gradient from a thermal step"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in packs.parameters_of("thermal")), \
        "scoped thermal pack received no gradient"
    assert all(p.grad is None for p in base_lm.parameters()), "base received gradient"
