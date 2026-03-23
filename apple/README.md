# MLX-TADA

TADA speech synthesis on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Setup

```bash
cd apple
uv venv
uv pip install -e ".[convert]"
```

For auto-transcription of reference audio (optional):
```bash
uv pip install mlx-whisper
```

## Convert Weights

Requires a [Hugging Face](https://huggingface.co/) account with access to `meta-llama/Llama-3.2-1B` (gated model). Login first:
```bash
huggingface-cli login
```

Then convert:
```bash
# 3B model
uv run python -m mlx_tada.convert_3b ./weights/3b

# 1B model
uv run python -m mlx_tada.convert_1b ./weights/1b
```

## Generate Speech

### CLI

```bash
uv run python -m mlx_tada.generate \
  --weights ./weights/3b \
  --audio speaker.wav \
  --text "The history of artificial intelligence is a fascinating journey that spans decades of research and innovation. It all began in the 1950s when pioneers like Alan Turing first posed the question of whether machines could think." \
  --output output.wav
```

With 4-bit quantization (10x faster, 60% less memory):
```bash
uv run python -m mlx_tada.generate \
  --weights ./weights/3b \
  --audio speaker.wav \
  --text "The history of artificial intelligence is a fascinating journey that spans decades of research and innovation. It all began in the 1950s when pioneers like Alan Turing first posed the question of whether machines could think." \
  --quantize 4 \
  --output output.wav
```

### Python

```python
from mlx_tada import TadaForCausalLM, save_wav

model = TadaForCausalLM.from_weights("./weights/3b", quantize=4)
ref = model.load_reference("speaker.wav")
out = model.generate("The history of artificial intelligence is a fascinating journey that spans decades of research and innovation. It all began in the 1950s when pioneers like Alan Turing first posed the question of whether machines could think.", ref)
save_wav(out.audio, "output.wav")

# out.audio     - numpy float32 array (24kHz)
# out.duration  - audio duration in seconds
# out.rtf       - real-time factor
# out.num_tokens
```

### Inference Options

Control generation behavior with `InferenceOptions`:

```python
from mlx_tada import TadaForCausalLM, InferenceOptions, save_wav

model = TadaForCausalLM.from_weights("./weights/3b", quantize=4)
ref = model.load_reference("speaker.wav")

opts = InferenceOptions(
    acoustic_cfg_scale=1.6,
    duration_cfg_scale=1.0,
    num_flow_matching_steps=10,
    time_schedule="logsnr",
    cfg_schedule="cosine",
)

out = model.generate(text="Hello world, today is a nice day.", reference=ref, inference_options=opts)
save_wav(out.audio, "output.wav")
```

The following inference options from the PyTorch version are **not currently supported** in MLX:
- `speed_up_factor`
- `num_acoustic_candidates`
- `scorer`
- `negative_condition_source`
- `text_only_logit_scale`
- `spkr_verification_weight`

### Speech Continuation

Use `num_extra_steps` to let the model generate speech beyond the provided text. The model continues speaking naturally and stops when it produces an end-of-sequence token:

```python
from mlx_tada import TadaForCausalLM, InferenceOptions, save_wav

model = TadaForCausalLM.from_weights("./weights/3b", quantize=4)
ref = model.load_reference("speaker.wav")

opts = InferenceOptions(
    acoustic_cfg_scale=1.6,
    num_flow_matching_steps=10,
    time_schedule="logsnr",
)

out = model.generate(
    text="The history of artificial intelligence is a fascinating journey that spans decades of research and innovation.",
    reference=ref,
    inference_options=opts,
    num_extra_steps=50,
)
save_wav(out.audio, "output.wav")
```

### Save and Reuse References

```python
from mlx_tada import Reference

ref = model.load_reference("speaker.wav")
ref.save("speaker.npz")

ref = Reference.load("speaker.npz")
out = model.generate("Reusing the same voice.", ref)
```

### Save Audio

```python
from mlx_tada import save_wav
save_wav(out.audio, "output.wav")
```

## Debug Logging

```bash
DEBUG=1 uv run python -m mlx_tada.generate \
  --weights ./weights/3b \
  --audio speaker.wav \
  --text "Hello"
```

```python
from mlx_tada import setup_logging

setup_logging()
```

## Running Tests

```bash
MLX_WEIGHTS=./weights/1b uv run pytest tests/ -v
```
