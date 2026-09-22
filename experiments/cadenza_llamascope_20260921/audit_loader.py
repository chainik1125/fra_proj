"""Independent release-code audit on real activations, separate from heldout tests."""
import ast
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))


def main():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import Config, MODEL_REVISIONS
    from train import capture_attention, prepare_data, tokenize
    from scope import ScopeSAE, sha256
    from remote import atomic_json
    torch.set_num_threads(8)
    campaign = HERE.parent
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=MODEL_REVISIONS['A'], token=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, revision=MODEL_REVISIONS['A'], token=False,
        torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval().requires_grad_(False)
    pools, _, _ = prepare_data(cfg)
    texts = [pools[k][i] for k in pools for i in range(2)]
    batch, valid = tokenize(tokenizer, texts, cfg.context_size)
    # Execute the two original published conversion functions, without importing
    # its distributed-training dependency tree or reimplementing those functions.
    source = (HERE / 'native_reference.py').read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SparseAutoEncoder')
    wanted = ('standardize_parameters_of_dataset_activation_scaling', 'transform_to_unit_decoder_norm')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    space = {'torch': torch, 'math': math, 'Dict': dict}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<pinned-native-conversions>', 'exec'), space)
    results = []
    with torch.inference_mode():
        for layer in (7, 11, 15):
            x_all = capture_attention(model, layer, batch, 'resid_post')
            x = x_all[valid].float()
            token_ids = batch['input_ids'][valid]
            positions = valid.nonzero()
            norm_x = x.norm(dim=-1)
            ordinary = norm_x <= 10 * norm_x.median()
            top_positions = norm_x.topk(5).indices
            torch.save({'x': x.cpu(), 'valid': valid.cpu(), 'input_ids':batch['input_ids'].cpu()},
                       campaign / f'loader_audit_L{layer:02d}_activations.pt')
            for expansion in (8, 32):
                run = campaign / f'scope-L{layer+1:02d}-{expansion}x-quality-a1'
                metadata = json.loads((run / 'sae_source.json').read_text())
                state = load_file(metadata['files']['weights']['path'], device='cuda')
                cfg_dict = metadata['release_config']
                sae = ScopeSAE(state, cfg_dict, layer)
                native_state = {k: v.float().clone() for k, v in state.items()}
                obj = SimpleNamespace(cfg=SimpleNamespace(**cfg_dict))
                for name in wanted:
                    native_state = space[name](obj, native_state)
                torch.testing.assert_close(sae.W_enc, native_state['encoder.weight'].T, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(sae.W_dec, native_state['decoder.weight'].T, atol=1e-7, rtol=1e-6)
                native_h = torch.nn.functional.linear(x, native_state['encoder.weight'], native_state['encoder.bias'])
                native_z = native_h.where(native_h > cfg_dict['jump_relu_threshold'], 0)
                native_recon = torch.nn.functional.linear(native_z, native_state['decoder.weight'], native_state['decoder.bias'])
                ours = sae(x)
                torch.testing.assert_close(ours, native_recon, atol=2e-3, rtol=1e-4)
                # SAELens 6.44's converter uses a scalar threshold before decoder
                # norm folding. Quantify the difference instead of assuming parity.
                scale = math.sqrt(4096) / cfg_dict['dataset_average_activation_norm']['in']
                h = torch.nn.functional.linear(x * scale, state['encoder.weight'].float(), state['encoder.bias'].float())
                z = h.where(h > cfg_dict['jump_relu_threshold'] * scale, 0)
                lens = torch.nn.functional.linear(z, state['decoder.weight'].float(), state['decoder.bias'].float()) / scale
                variance = (x - x.mean(0)).square().sum()
                per_token_sse = (ours - x).square().sum(-1)
                row = {'residual_layer': layer, 'expansion': expansion, 'tokens': len(x),
                    'native_parameters_close': True, 'parameter_atol':1e-7, 'parameter_rtol':1e-6,
                    'native_max_abs_error': float((ours-native_recon).abs().max()),
                    'native_reference_sha256': sha256(HERE / 'native_reference.py'),
                    'native_fvu': float(per_token_sse.sum()/variance),
                    'saelens_converter_fvu': float((lens-x).square().sum()/variance),
                    'native_l0': float((native_z>0).sum(-1).float().mean()),
                    'saelens_converter_l0': float((z>0).sum(-1).float().mean()),
                    'x_norm_quantiles': x.norm(dim=-1).quantile(torch.tensor([0., .5, .99, 1.],device=x.device)).tolist(),
                    'recon_norm_quantiles': ours.norm(dim=-1).quantile(torch.tensor([0., .5, .99, 1.],device=x.device)).tolist(),
                    'largest_1pct_sse_fraction': float(per_token_sse.topk(max(1,len(x)//100)).values.sum()/per_token_sse.sum()),
                    'posthoc_ordinary_token_filter':'activation norm <= 10 times median; diagnostics only, steering unchanged',
                    'ordinary_tokens':int(ordinary.sum()),
                    'ordinary_fvu':float(per_token_sse[ordinary].sum() / (x[ordinary]-x[ordinary].mean(0)).square().sum()),
                    'largest_norm_tokens':[{'example':int(positions[i,0]), 'padded_position':int(positions[i,1]),
                         'token_id':int(token_ids[i]), 'decoded_token':tokenizer.decode([int(token_ids[i])]),
                         'activation_norm':float(norm_x[i])} for i in top_positions]}
                results.append(row)
                atomic_json(campaign / 'loader_audit.json', {'state':'running','results':results})
                del sae, state, native_state, h, z, lens, ours, native_h, native_z, native_recon
    atomic_json(campaign / 'loader_audit.json', {'state':'complete','results':results,
        'data':'first two training examples per class, no validation/test selection',
        'reference':'OpenMOSS/Llamascopium b932639261697c0642d077c05f2cb7e463d058cb'})


if __name__ == '__main__':
    main()
