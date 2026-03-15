# test_encoder_removal.py
# Run: srun --partition=eval --gres=gpu:1 python test_encoder_removal.py

import torch
import torchaudio
import time
import os

from tada.modules.encoder import Encoder, EncoderOutput
from tada.modules.tada import TadaForCausalLM, InferenceOptions
from tada.utils.test_utils import get_sample_dir

def cast_prompt(prompt, dtype):
    """Cast all tensor fields in EncoderOutput to the given dtype."""
    from dataclasses import replace
    kwargs = {}
    for f in prompt.__dataclass_fields__:
        v = getattr(prompt, f)
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            kwargs[f] = v.to(dtype)
    return replace(prompt, **kwargs)

device = "cuda"
torch.cuda.reset_peak_memory_stats()

# ──────────────────────────────────────────────
# 1. Load model (encoder NOT loaded anymore)
# ──────────────────────────────────────────────
print("=" * 60)
print("1. Loading TadaForCausalLM (no encoder inside)")
print("=" * 60)

mem_before_model = torch.cuda.max_memory_allocated() / 1e9
model = TadaForCausalLM.from_pretrained("HumeAI/tada-3b-ml", torch_dtype=torch.bfloat16).to(device)
# Decoder uses weight-norm and needs fp32; reload it after .to()
model._decoder = model._decoder.float().to(device)

mem_after_model = torch.cuda.max_memory_allocated() / 1e9
print(f"   Peak GPU memory after model load: {mem_after_model:.2f} GB")
print(f"   Has _tokenizer: {hasattr(model, '_tokenizer')}")
print(f"   Has _encoder:   {hasattr(model, '_encoder')}")
print(f"   Tokenizer type: {type(model.tokenizer).__name__}")

# ──────────────────────────────────────────────
# 2. Load encoder separately
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Loading Encoder separately")
print("=" * 60)

encoder = Encoder.from_pretrained("HumeAI/tada-codec").to(device)

mem_after_encoder = torch.cuda.max_memory_allocated() / 1e9
print(f"   Peak GPU memory after encoder load: {mem_after_encoder:.2f} GB")
print(f"   Encoder cost: ~{mem_after_encoder - mem_after_model:.2f} GB")

# ──────────────────────────────────────────────
# 3. Load reference audio
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Loading reference audio")
print("=" * 60)

audio_path = os.path.join(get_sample_dir(), "fb_ears/fb_ears_emo_amusement_freeform.wav")
audio, sr = torchaudio.load(audio_path)
audio = audio.mean(0, keepdim=True).to(device)
audio = audio / audio.abs().max()
print(f"   Audio length: {audio.shape[-1] / sr:.1f}s")

# ──────────────────────────────────────────────
# 4. Generate WITHOUT prompt cache (includes encoding)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Generate (no cache — includes encoding time)")
print("=" * 60)

test_text = "This is a test of the encoder removal optimization."

t0 = time.time()
prompt = encoder(audio, sample_rate=sr)
t_encode = time.time()
prompt = cast_prompt(prompt, torch.bfloat16)
output = model.generate(prompt=prompt, text=test_text)
t_generate = time.time()

gen_duration = output.audio[0].shape[-1] / 24000
total_time = t_generate - t0
encode_time = t_encode - t0
generate_time = t_generate - t_encode

print(f"   Encode time:    {encode_time:.2f}s")
print(f"   Generate time:  {generate_time:.2f}s")
print(f"   Total time:     {total_time:.2f}s")
print(f"   Audio length:   {gen_duration:.2f}s")
print(f"   RTF (total):    {total_time / gen_duration:.3f}x")
print(f"   RTF (gen only): {generate_time / gen_duration:.3f}x")

torchaudio.save("test_no_cache.wav", output.audio[0].cpu().float().unsqueeze(0), 24000)
print("   Saved: test_no_cache.wav")

# ──────────────────────────────────────────────
# 5. Prompt caching round-trip
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Prompt caching round-trip")
print("=" * 60)

prompt.save("prompt_cache.pt")
cache_size = os.path.getsize("prompt_cache.pt") / 1e6
print(f"   Cache file size: {cache_size:.1f} MB")

prompt_cached = EncoderOutput.load("prompt_cache.pt", device=device)

for field in prompt.__dataclass_fields__:
    orig = getattr(prompt, field)
    cached = getattr(prompt_cached, field)
    if isinstance(orig, torch.Tensor):
        match = torch.allclose(orig, cached)
        print(f"   {field:30s} match={match}  shape={tuple(orig.shape)}")
    else:
        match = orig == cached
        print(f"   {field:30s} match={match}")

# ──────────────────────────────────────────────
# 6. Generate WITH prompt cache (no encoding)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Generate (from cached prompt — no encoding)")
print("=" * 60)

t0 = time.time()
prompt_cached = cast_prompt(prompt_cached, torch.bfloat16)
output_cached = model.generate(prompt=prompt_cached, text=test_text)
t1 = time.time()

gen_duration_cached = output_cached.audio[0].shape[-1] / 24000
wall_time_cached = t1 - t0

print(f"   Total time:     {wall_time_cached:.2f}s")
print(f"   Audio length:   {gen_duration_cached:.2f}s")
print(f"   RTF (total):    {wall_time_cached / gen_duration_cached:.3f}x")

torchaudio.save("test_from_cache.wav", output_cached.audio[0].cpu().float().unsqueeze(0), 24000)
print("   Saved: test_from_cache.wav")

# ──────────────────────────────────────────────
# 7. Memory comparison
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Memory summary")
print("=" * 60)

del encoder
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

output3 = model.generate(prompt=cast_prompt(prompt_cached, torch.bfloat16), text=test_text)
mem_no_encoder = torch.cuda.max_memory_allocated() / 1e9

print(f"   Model load (no encoder):                {mem_after_model:.2f} GB")
print(f"   Model load + encoder:                   {mem_after_encoder:.2f} GB")
print(f"   Encoder overhead:                       ~{mem_after_encoder - mem_after_model:.2f} GB")
print(f"   Peak during generate (no encoder):      {mem_no_encoder:.2f} GB")

# ──────────────────────────────────────────────
# 8. RTF comparison
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. RTF comparison")
print("=" * 60)
print(f"   Without cache:  {total_time / gen_duration:.3f}x RTF  (encode={encode_time:.2f}s + generate={generate_time:.2f}s)")
print(f"   With cache:     {wall_time_cached / gen_duration_cached:.3f}x RTF  (generate only)")
print(f"   Speedup:        {total_time / wall_time_cached:.2f}x")

print("\n" + "=" * 60)
print("DONE — listen to test_no_cache.wav and test_from_cache.wav")
print("=" * 60)

# Cleanup
os.remove("prompt_cache.pt")
