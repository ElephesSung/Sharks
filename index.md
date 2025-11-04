---
layout: default
---

<!-- Load MathJax -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

## Two parameters: $\mu_{\lambda}$ & $\sigma_{\lambda}$ — Mean and STD of $\lambda$ in the Poisson killing model

**Mathematical model**: Per-cell Poisson Model (Individual Kill Counts)

- Each cell \( i \in \{1, ..., M\} \) has a latent killing rate \( \lambda_i \).
- This rate is drawn from a Gamma distribution parameterized by \( \mu_{\lambda} \) and \( \sigma_{\lambda} \).
- The cell’s observed kill count \( k_i \) is Poisson-distributed with rate \( \lambda_i \).

### Layers in the model:

1. \( \lambda_i \sim \mathrm{Gamma}(\alpha, \beta) \)

   Mean and standard deviation:

   $$
   \mu_{\lambda} = \frac{\alpha}{\beta}
   $$

   $$
   \sigma_{\lambda} = \frac{\sqrt{\alpha}}{\beta}
   $$

2. \( N_j \sim \mathrm{Poisson}(\lambda_i) \)
