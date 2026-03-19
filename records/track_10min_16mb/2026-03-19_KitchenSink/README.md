# KitchenSink — 2026-03-19

Combines all proven winning techniques from the top two PRs (#88, #89) plus novel edges.

## Techniques

### Proven (from top PRs)
- **Int6 quantization + zstd-22**: Per-row amax scaling with [-31,31] range, zstd level 22 compression
- **MLP 3x expansion**: 1536 hidden dim, enabled by int6 artifact savings
- **FP16 tied embedding passthrough**: Keeps tok_emb.weight in fp16 during export
- **NorMuon optimizer**: Row-normalized Newton-Schulz orthogonalization with second-moment tracking
- **MTP auxiliary head**: Predicts token i+2 from hidden state i, 0.1x loss weight, stripped from artifact
- **In-place QAT**: Last 15% of training, fake-quantize weights in-place before forward, restore after backward
- **SWA**: Checkpoint averaging during warmdown phase
- **EMA**: Exponential moving average tracking (decay=0.999), used when SWA unavailable
- **Batched sliding window eval**: stride=64, batch=32 windows, forward_logits() method

### Novel Edges
- **Progressive context**: Train at seq_len=1024 for first 70%, switch to seq_len=4096 for last 30%
- **LoRA TTT**: Test-time training with rank-4 LoRA on Q/V projections (optional, disabled by default)

## Architecture
- 9 transformer blocks, 512 dim, 8 heads, 4 KV heads
- MLP hidden = 1536 (3x), tied embeddings, vocab 1024
- Skip connections (encoder-decoder pattern)

## Usage
```bash
# 8xH100 full run
torchrun --nproc_per_node=8 train_gpt.py

# Quick smoke test
MAX_WALLCLOCK_SECONDS=120 python train_gpt.py
```

## Dependencies
Requires `zstandard` for zstd compression (falls back to zlib if unavailable).
