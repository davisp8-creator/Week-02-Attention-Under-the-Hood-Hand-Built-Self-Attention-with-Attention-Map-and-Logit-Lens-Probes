# Week-02-Attention-Under-the-Hood-Hand-Built-Self-Attention-with-Attention-Map-and-Logit-Lens-Probes.
Hand-Built Self-Attention with Attention-Map and Logit-Lens Probes

## Results

Both positional-encoding variants were trained on the copy task for 1000 steps on CUDA and converged nearly identically, from an initial loss of ~3.15–3.18 down to 0.0012 by step 800. On this short-sequence copy task, learned and sinusoidal encodings are effectively interchangeable: both produce the clean diagonal attention pattern and confident logit-lens predictions seen in the probes below, including on the repeats, ascending, and constant input variants.

### Training Loss

| Step | Learned | Sinusoidal |
|-----:|--------:|-----------:|
| 0    | 3.1541  | 3.1821     |
| 200  | 0.0129  | 0.0135     |
| 400  | 0.0039  | 0.0038     |
| 600  | 0.0019  | 0.0020     |
| 800  | 0.0012  | 0.0012     |

<details>
<summary>Full training log — learned positional encoding</summary>

```text
=== Positional encoding: learned ===
Training model on copy task (cuda)...
Step 0 | Loss: 3.1541
Step 200 | Loss: 0.0129
Step 400 | Loss: 0.0039
Step 600 | Loss: 0.0019
Step 800 | Loss: 0.0012
Training complete.
Attention map saved to results/attention_maps_learned.png
Logit lens saved to results/logit_lens_learned.png
Attention map saved to results/attention_maps_learned_repeats.png
Logit lens saved to results/logit_lens_learned_repeats.png
Attention map saved to results/attention_maps_learned_ascending.png
Logit lens saved to results/logit_lens_learned_ascending.png
Attention map saved to results/attention_maps_learned_constant.png
Logit lens saved to results/logit_lens_learned_constant.png
```

</details>

<details>
<summary>Full training log — sinusoidal positional encoding</summary>

```text
=== Positional encoding: sinusoidal ===
Training model on copy task (cuda)...
Step 0 | Loss: 3.1821
Step 200 | Loss: 0.0135
Step 400 | Loss: 0.0038
Step 600 | Loss: 0.0020
Step 800 | Loss: 0.0012
Training complete.
Attention map saved to results/attention_maps_sinusoidal.png
Logit lens saved to results/logit_lens_sinusoidal.png
Attention map saved to results/attention_maps_sinusoidal_repeats.png
Logit lens saved to results/logit_lens_sinusoidal_repeats.png
Attention map saved to results/attention_maps_sinusoidal_ascending.png
Logit lens saved to results/logit_lens_sinusoidal_ascending.png
Attention map saved to results/attention_maps_sinusoidal_constant.png
Logit lens saved to results/logit_lens_sinusoidal_constant.png
```

</details>

### Interpretability Probes

#### Learned positional encoding

![Attention maps (learned)](results/attention_maps_learned.png)

![Logit lens (learned)](results/logit_lens_learned.png)

<details>
<summary>Additional input variants (repeats, ascending, constant)</summary>

**Repeats**

![Attention maps (learned, repeats)](results/attention_maps_learned_repeats.png)

![Logit lens (learned, repeats)](results/logit_lens_learned_repeats.png)

**Ascending**

![Attention maps (learned, ascending)](results/attention_maps_learned_ascending.png)

![Logit lens (learned, ascending)](results/logit_lens_learned_ascending.png)

**Constant**

![Attention maps (learned, constant)](results/attention_maps_learned_constant.png)

![Logit lens (learned, constant)](results/logit_lens_learned_constant.png)

</details>

#### Sinusoidal positional encoding

![Attention maps (sinusoidal)](results/attention_maps_sinusoidal.png)

![Logit lens (sinusoidal)](results/logit_lens_sinusoidal.png)

<details>
<summary>Additional input variants (repeats, ascending, constant)</summary>

**Repeats**

![Attention maps (sinusoidal, repeats)](results/attention_maps_sinusoidal_repeats.png)

![Logit lens (sinusoidal, repeats)](results/logit_lens_sinusoidal_repeats.png)

**Ascending**

![Attention maps (sinusoidal, ascending)](results/attention_maps_sinusoidal_ascending.png)

![Logit lens (sinusoidal, ascending)](results/logit_lens_sinusoidal_ascending.png)

**Constant**

![Attention maps (sinusoidal, constant)](results/attention_maps_sinusoidal_constant.png)

![Logit lens (sinusoidal, constant)](results/logit_lens_sinusoidal_constant.png)

</details>

