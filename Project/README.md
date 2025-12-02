## Combined NEWS DJIA


### As features vector is (date, label, top1, top2, ...., top25)

## $\rightarrow$ **find a way to weight news**

### Solution 1: 

$$\text{Vector}_{\text{Text}} = \text{Transformer}(\text{Text})$$

$$\text{Combined}_{\text{Vector}} = \text{Concatenate}(\text{Vector}_{\text{Text}}, \text{Importance})$$
$$\text{Prediction} = \text{Dense}(\text{Combined}_{\text{Vector}}) \rightarrow \{0, 1\}$$


### Solution 2:

- **Concept:** Encode the text for all 25 news items for a given day into a vector. Then, use the importance score (Top $i$) to create a **weighted average** of these 25 vectors.

- **Process:**
1. Get the vector embedding for News 1 ($V_1$), News 2 ($V_2$), ..., News 25 ($V_{25}$).
2. Use the normalized importance scores ($I_1, I_2, ..., I_{25}$) as weights.
3. Calculate the final Daily News Vector ($V_{D}$):
$$V_{D} = \frac{\sum_{i=1}^{25} I_i \cdot V_i}{\sum_{i=1}^{25} I_i}$$

- **Advantage**: This creates a single, highly condensed feature vector ($V_{D}$) for the entire day that inherently prioritizes the most important news. This $V_D$ is then used as the input feature for your time series or classification model.