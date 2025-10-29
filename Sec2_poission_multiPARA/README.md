# Second Section: multi-parameter poission process with Bayesian Inference

For the first section, we only have one parameter, which is the killing rate, for the mathematical model of NK cell-mediated cytotoxicity. Here we introduce multiple parameters to further study the homogenenous and heterogenous of the NK cell population and how the environmental variation change the NK cell cytotoxicity.


Here, we introduce a the framwork with multiple parameters to account for: **Heterogeneity** within the NK cell population.




## Scenario-1. 2 parameters: $\mu_{\lambda}$ & $\sigma_{\lambda}$ ---  Mean and STD of $\lambda$ in the Poisson killing model

**Mathematical model 1**: Per-cell Poisson Model (Individual Kill Counts)

- Each cell $i \in \{1, \dots, M\}$  has a latent killing rate $\lambda_i$.
- This rate is drawn from a population distribution parameterised by $\mu_\lambda $  and $ \sigma_\lambda $
- The cell’s observed kill count $ k_i $ is Poisson-distributed with rate $ \lambda_i $

<br>

**Mathematical model 2**: Marginalised Multinomial Model (Histogram Data)

- Only the histogram of kill counts is observed, not the identity of individual cells
- The individual cell rates $ \lambda $ follow a population distribution as above
- The Poisson likelihood is marginalised over $ \lambda $



---
Next step:
- Check the code, model 1 and model 2 should be the same
- play with different cell number in synthetic data
- play with different distributions, different prior distributions
- The posterior distribution check
- add p_zero
- Get a detailed model for the contact and kill history
---
