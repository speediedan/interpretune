"""Decoder-space concept-mass splitting probe: 16k vs 262k gemma-scope-2-1b-it transcoders.

For a matching layer, project each feature's decoder vector onto the Fruit-Color logit-diff
direction (unembed cols of the best-variant token ids) and compare mass concentration across
widths (top-k share, participation ratio). Pure tensor math - no model forward needed.

All work happens in ``main()`` so importing this module is side-effect-free — the test suite runs
pytest with ``--doctest-modules``, which imports every module under ``tests/`` during collection.
"""

import json

LAYER = 19  # matches the curated clear-fruit feature's layer (19/11234)
REPO = "mwhanna/gemma-scope-2-1b-it"


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16)
    W_U = model.get_output_embeddings().weight.detach().float()  # (vocab, d_model)
    del model

    fruit_ids = [tok.encode(v, add_special_tokens=False)[0] for v in ("Fruit", " Fruit")]
    color_ids = [tok.encode(v, add_special_tokens=False)[0] for v in ("Color", " Color")]
    diff_dir = W_U[fruit_ids].mean(0) - W_U[color_ids].mean(0)  # (d_model,)
    diff_dir = diff_dir / diff_dir.norm()

    results = {}
    for width_name, subdir in (("16k", "width_16k_l0_small_affine"), ("262k", "width_262k_l0_small_affine")):
        path = hf_hub_download(REPO, f"transcoder_all/{subdir}/layer_{LAYER}.safetensors")
        sd = load_file(path)
        dec_key = next(k for k in sd if "W_dec" in k or "w_dec" in k.lower())
        W_dec = sd[dec_key].float()  # expect (n_features, d_model)
        if W_dec.shape[0] < W_dec.shape[1]:
            W_dec = W_dec.T
        proj = (W_dec @ diff_dir).abs()  # (n_features,)
        mass = proj.sum()
        sorted_proj, _ = torch.sort(proj, descending=True)
        p = proj / mass
        participation_ratio = float(1.0 / (p.pow(2).sum()))
        results[width_name] = {
            "n_features": int(proj.numel()),
            "total_abs_projection_mass": float(mass),
            "top1_share": float(sorted_proj[0] / mass),
            "top10_share": float(sorted_proj[:10].sum() / mass),
            "top100_share": float(sorted_proj[:100].sum() / mass),
            "top1000_share": float(sorted_proj[:1000].sum() / mass),
            "participation_ratio": participation_ratio,
            "participation_fraction": participation_ratio / proj.numel(),
            "max_abs_projection": float(sorted_proj[0]),
        }
        del sd, W_dec, proj

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
