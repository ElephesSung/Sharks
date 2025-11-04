# Second Section: multi-parameter poission process with Bayesian Inference

For the first section, we only have one parameter, which is the killing rate, for the mathematical model of NK cell-mediated cytotoxicity. Here we introduce multiple parameters to further study the homogenenous and heterogenous of the NK cell population and how the environmental variation change the NK cell cytotoxicity.


Here, we introduce a the framwork with multiple parameters to account for: **Heterogeneity** within the NK cell population.




## Two parameters: $\mu_{\lambda}$ & $\sigma_{\lambda}$ — Mean and STD of $\lambda$ in the Poisson killing model

**Mathematical model**: Per-cell Poisson Model (Individual Kill Counts)

- Each cell $i \in \{1, \dots, M\}$ has a latent killing rate $\lambda_i$.
- This rate is drawn from a Gamma distribution parameterized by $\mu_{\lambda}$ and $\sigma_{\lambda}$.
- The cell’s observed kill count $k_i$ is Poisson-distributed with rate $\lambda_i$.

Therefore, there are two layers in this model:

1. $\lambda_i$ for each cell follows the Gamma distribution:

   Mean and standard deviation are given by:

   ```math
   \mu_{\lambda} = \frac{\alpha}{\beta}, \sigma_{\lambda} = \frac{\sqrt{\alpha}}{\beta}
   ```

   ```math
   \lambda_i \sim \mathrm{Gamma}(\alpha, \beta)
   ```

2. $N_j$ (the number of target cells killed per killer cell) follows the Poisson distribution with rate $\lambda_i$:

   $$
   N_j \sim \mathrm{Poisson}(\lambda_i)
   $$



