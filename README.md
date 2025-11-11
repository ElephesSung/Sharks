# Sharks

<!-- Test whether the NK cell cytotoxic decision-making is history-dependent or not


   ```math
   \mu_{\lambda} = \frac{\alpha}{\beta}, \sigma_{\lambda} = \frac{\sqrt{\alpha}}{\beta}
   ```

   ```math
   \lambda_i \sim \mathrm{Gamma}(\alpha, \beta)
   ``` -->


---

## 1. one parameter model: Poisson distribution -- $\lambda$
```math
p(N|N_k) ={\frac {\lambda ^{k}e^{-\lambda }}{k!}}
```
$N$ is the number of the ***target cells killed per kill cell***.


<br>

---

### **Synthetic Data & Parameter Inference**

- 100 cells.
![100](./Sec1_poission/syn_results/1P_100-001.png)

- 250 cells.
![250](./Sec1_poission/syn_results/1P_250-001.png)

- 500 cells.
![500](./Sec1_poission/syn_results/1P_500-001.png)

- 1000 cells.
![1000](./Sec1_poission/syn_results/1P_1000-001.png)

### **Inference for experimental data**

1. **NKG2D Data**
   > Only the `NK-WT VS RMA-MULT` and `NK-WT VS RMA-RAE` data can be extracted. For other conditions, further confirmation is required. 

   **Experimental data visualisation**:
   ![experimenta_data](./Expe_2/one_p/plots/experiment.png)

   **Parameter inference**:
   ![inference_1P](./Expe_2/one_p/plots/rate_posteriors_facet.png)

   **Posterior prediction**:
   ![inference_1P](./Expe_2/one_p/plots/posterior_violin_NK_VS_RMA-MULT-001.png)
   ![inference_1P](./Expe_2/one_p/plots/posterior_violin_NK_VS_RMA-RAE-001.png)

<br>

2. Rtx and bispecific data

   coming

   ![coming](./patrick-star.gif)



<br>
<br>

---

## 2. two parameter model: Zero-inflated Poisson distribution. -- $\lambda$ & $p_z$
```math
P(N = k \mid \lambda, p_z) =
\begin{cases}
p_z + (1 - p_z)e^{-\lambda}, & k = 0, \\[8pt]
(1 - p_z)\dfrac{e^{-\lambda}\lambda^k}{k!}, & k > 0~.
\end{cases}
```

### Synthetics data and inference for test

- 100 cells
![100](./Sec3_poission_zeroP/syn_results/pngs/ZIP_100-001.png)

- 250 cells
![250](./Sec3_poission_zeroP/syn_results/pngs/ZIP_250-001.png)

- 500 cells
![500](./Sec3_poission_zeroP/syn_results/pngs/ZIP_500-001.png)

- 1000 cells
![1000](./Sec3_poission_zeroP/syn_results/pngs/ZIP_1000-001.png)

- 5000 cells
![5000](./Sec3_poission_zeroP/syn_results/pngs/ZIP_5000-001.png)

- 50000 cells
![50000](./Sec3_poission_zeroP/syn_results/pngs/ZIP_50000-001.png)


### **Inference for experimental data**
![coming](./patrick-star.gif)




<br>
<br>

---

## 3. two parameter model: Gamma distribution + Poisson distribution. -- $\mu_{\lambda}$ & $\sigma_{\lambda}$

```math
P(N = k \mid \lambda, p_z) =
\begin{cases}
p_z + (1 - p_z)e^{-\lambda}, & k = 0, \\[8pt]
(1 - p_z)\dfrac{e^{-\lambda}\lambda^k}{k!}, & k > 0~.
\end{cases}
```