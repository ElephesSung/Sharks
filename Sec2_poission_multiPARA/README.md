# Second Section: multi-parameter poission process with Bayesian Inference

For the first section, we only have one parameter, which is the killing rate, for the mathematical model of NK cell-mediated cytotoxicity. Here we introduce multiple parameters to further study the homogenenous and heterogenous of the NK cell population and how the environmental variation change the NK cell cytotoxicity.


Here, we introduce a the framwork with multiple parameters to account for: **Heterogeneity** within the NK cell population.




## Two parameters: $\mu_{\lambda}$ & $\sigma_{\lambda}$ — Mean and STD of $\lambda$ in the Poisson killing model

### Mathematical model
Per-cell Poisson Model (Individual Kill Counts)

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

   ```math
   N_j \sim \mathrm{Poisson}(\lambda_i)
   ```


### Bayes' theorem & Bayesian Inference

```math
P(\mu_{\lambda}, \sigma_{\lambda} | N_j) = \frac{P(\mu_{\lambda}, \sigma_{\lambda}) \cdot P(N_j | \mu_{\lambda}, \sigma_{\lambda})}{P(N_j)}
```

For the likelihood:
```math
P(N \mid \mu_\lambda, \sigma_\lambda)
= \prod_{j=1}^{M} P(N_j \mid \mu_\lambda, \sigma_\lambda),
```

```math
\begin{aligned}
P(N_j | \mu_{\lambda}, \sigma_{\lambda}) 
&=  \int_{0}^{ +\infty } P(N_j | \lambda_i) \cdot P(\lambda_i | \mu_{\lambda}, \sigma_{\lambda})\mathrm{d}\lambda_i \\
&=  \int_{0}^{ +\infty } (\frac{\lambda_i^{N_j}e^{-\lambda_i}}{N_j !})\cdot (\frac{\beta^{\alpha}}{\Gamma(\alpha)}\lambda_i^{\alpha-1}e^{-\beta\lambda_i})\mathrm{d}\lambda_i \\
&= \frac{\beta^{\alpha}}{\Gamma(\alpha)N_j !} \int_{0}^{ +\infty}\lambda_i^{N_j +\alpha -1} e^{-\lambda_i(\beta +1)} \mathrm{d}\lambda_i
\end{aligned}
```

**Analytically**:
```math
\begin{aligned}
\frac{\beta^{\alpha}}{\Gamma(\alpha)N_j !} \int_{0}^{ +\infty}\lambda_i^{N_j +\alpha -1} e^{-\lambda_i(\beta +1)} \mathrm{d}\lambda_i
&=  \frac{\Gamma(\alpha + N_j)}{\Gamma(\alpha)\, N_j!}
\left(\frac{\beta}{\beta + 1}\right)^{\alpha}
\left(\frac{1}{\beta + 1}\right)^{N_j}
\end{aligned} \\

\Rightarrow P(N_j | \mu_{\lambda}, \sigma_{\lambda}) = \mathrm{NegBin}(r= \alpha, p = \frac{\beta}{\beta +1})

```




**Numerically**, we can use [Gauss–Laguerre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature):

```math
{\displaystyle \int _{0}^{+\infty }e^{-x}f(x)\,dx\approx \sum _{i=1}^{n}w_{i}f(x_{i})}
```
<br>

Therefore:

```math

\begin{aligned}
p(N_j \mid \mu_\lambda, \sigma_\lambda)
&=
\frac{\beta^{\,\alpha}}{\Gamma(\alpha)\,N_j!}
\int_{0}^{+\infty}
\lambda_i^{\,N_j+\alpha-1}
e^{-(\beta+1)\lambda_i}\,
\mathrm{d}\lambda_i
\\[6pt]
&=
\frac{\beta^{\,\alpha}}
{\Gamma(\alpha)\,N_j!\,(\beta+1)^{\,N_j+\alpha}}
\int_{0}^{+\infty} e^{-x} x^{N_j+\alpha-1} \mathrm{d}x
\\[6pt]
&\;\approx\;
\frac{\beta^{\,\alpha}}
{\Gamma(\alpha)\,N_j!\,(\beta+1)^{\,N_j+\alpha}}
\sum_{k=1}^{n_\text{quad}}
w_k\,
(x_k)^{N_j+\alpha-1}
\end{aligned}
```
$n_\text{quad}$: Represents the number of points used in the quadrature approximation. A higher value generally leads to a more accurate approximation.
$x_k$: Are the nodes (or abscissas) of the Gauss-Laguerre quadrature. These are the roots of the $n_\text{quad}$-th Laguerre polynomial, denoted as $L_{n_\text{quad}}(x)$.
$w_k$: Are the corresponding weights for each node. The weight $w_k$ associated with the node $x_k$ is calculated as: $w_k = \frac{x_k}{(n_\text{quad}+1)^2 [L_{n_\text{quad}+1}(x_k)]^2}$

