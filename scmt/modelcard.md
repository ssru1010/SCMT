# Schema-Augmented SOMT (W)


**Summary**
A compact, self-contained PyTorch implementation of Schema-Augmented SOMT — a memory-augmented Transformer with a persistent schema layer implemented as global buffers. The package exposes HF-style convenience functions `from_pretrained()` and `save_pretrained()` and a `generate()` helper for sampling.


**Usage**
```python
from somt import SchemaAugmentedSOMT, SOMTConfig
model = SchemaAugmentedSOMT(SOMTConfig(vocab_size=50257).vocab_size)
# or load a saved model
model = SchemaAugmentedSOMT.from_pretrained("/path/to/checkpoint")
```


**License & Authors**
Add your preferred license and authorship information here.