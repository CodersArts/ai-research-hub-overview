"""
generate.py — Inference: load model and generate Python code
=============================================================
Run:
  python src/generate.py --prompt "read a csv file"
  python src/generate.py                             # interactive REPL
"""
import argparse, torch
from tokenizer import CharTokenizer
from model import TinyGPT


def load(model_path="model.pt", tok_path="tokenizer.json"):
    tok = CharTokenizer()
    tok.load(tok_path)
    ckpt  = torch.load(model_path, map_location="cpu")
    cfg   = ckpt["config"]
    model = TinyGPT(**cfg, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model (loss={ckpt['loss']:.4f})")
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt, max_new=200, temperature=0.8, top_k=40):
    """Generate code from a natural language prompt."""
    ids = tok.encode(prompt + tok.SEP_TOKEN)
    ctx = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_new):
        x      = ctx[:, -model.ctx_len:]
        logits = model(x)[:, -1, :] / temperature

        # Top-k: zero out all but top-k logits
        if top_k > 0:
            cutoff = torch.topk(logits, top_k).values[:, -1:]
            logits[logits < cutoff] = float("-inf")

        probs   = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ctx     = torch.cat([ctx, next_id], dim=1)

        # Stop at second SEP token
        if next_id.item() == tok.sep_id and ctx.shape[1] > len(ids) + 5:
            break

    full = tok.decode(ctx[0].tolist())
    code = full.split(tok.SEP_TOKEN, 1)[-1].split(tok.SEP_TOKEN)[0].strip()
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=200)
    args = parser.parse_args()

    model, tok = load()

    if args.prompt:
        print(generate(model, tok, args.prompt, args.max_tokens, args.temperature))
    else:
        print("Python Code Generator | type 'quit' to exit\n")
        while True:
            try:
                p = input("Describe what you want: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if p.lower() in ("quit", "exit", ""):
                break
            print("\n" + "─"*40)
            print(generate(model, tok, p))
            print("─"*40 + "\n")
