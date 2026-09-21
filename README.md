# Week-02-Attention-Under-the-Hood-Hand-Built-Self-Attention-with-Attention-Map-and-Logit-Lens-Probes.
Hand-Built Self-Attention with Attention-Map and Logit-Lens Probes

## Results

Both positional-encoding variants were trained on the copy task for 1000 steps on a Colab T4 (CUDA). Each converged to near-zero loss.

### Training Loss

| Step | Learned | Sinusoidal |
|-----:|--------:|-----------:|
| 0    | 3.1541  | 3.1821     |
| 200  | 0.0129  | 0.0135     |
| 400  | 0.0039  | 0.0038     |
| 600  | 0.0019  | 0.0020     |
| 800  | 0.0012  | 0.0012     |

### Interpretability Probes

**Learned positional encoding**

![Attention maps (learned)](results/attention_maps_learned.png)

![Logit lens (learned)](results/logit_lens_learned.png)

**Sinusoidal positional encoding**

![Attention maps (sinusoidal)](results/attention_maps_sinusoidal.png)

![Logit lens (sinusoidal)](results/logit_lens_sinusoidal.png)

