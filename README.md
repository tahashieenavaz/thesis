# Margin-Enhanced Logit Normalization Loss

- [Margin-Enhanced Logit Normalization Loss](#margin-enhanced-logit-normalization-loss)
  - [Introduction](#introduction)
  - [Approach](#approach)
    - [Python](#python)
    - [Matlab](#matlab)
  - [Dataset](#dataset)
    - [Sample Images](#sample-images)
  - [References](#references)

## Introduction

Out-of-distribution (OOD) detection is a critical aspect of deep learning, especially in deploying models for real-world applications. Deep learning models are trained on specific datasets and are often highly specialized in recognizing patterns within the distribution of their training data. However, when presented with inputs that fall outside this distribution, such as novel or anomalous data points, these models may produce unreliable predictions with high confidence, leading to catastrophic outcomes in sensitive domains like healthcare, autonomous driving, and financial systems. Therefore, the ability to accurately identify OOD samples ensures that the system remains robust and dependable by flagging cases that require additional human intervention or alternative processing strategies.

From an academic standpoint, OOD detection also addresses broader challenges in machine learning, such as enhancing generalization and mitigating overfitting. Robust detection mechanisms facilitate the safe deployment of models in dynamic environments where the data encountered can vary significantly from the training set. Furthermore, the integration of OOD detection frameworks allows researchers to explore the boundaries of a model's epistemic uncertainty, which is crucial for developing models with calibrated confidence levels. This not only promotes transparency and interpretability in machine learning systems but also aligns with the ethical imperative to mitigate potential risks associated with automation and decision-making in AI-driven technologies.

## Approach

$$
L_{MultiClassHinge}[\frac{logits}{(|logits + ReLU(\theta)|_2 + \epsilon)} * \phi, targets]
$$

<br />

The **Margin-Enhanced LogitNorm Loss (MELN)** is mathematically implemented by first computing the Euclidean norm ($L^2$ Norm) of the input vectors $x$, denoted as $\left\| x \right\|_2$, and ensuring numerical stability by adding a small constant ($10^7$).

The logits are normalized by subtracting a learnable margin parameter ($m$) and dividing the result by the norm. This operation enhances the decision boundary by introducing a margin between the logits. The normalized logits are then scaled by a learnable temperature parameter ($T$) to control the sharpness of the logits.

The final step involves applying the cross-entropy loss function on these margin-enhanced, temperature-scaled normalized logits and the target labels, promoting discriminative representations while maintaining the desired separation in the feature space.

### Python

```python
class MarginTemperatureEnhancedHingeLoss(torch.nn.Module):
    def __init__(self, margin: float = 1.0, tradeoff: float = 0.5):
        super(MarginTemperatureEnhancedHingeLoss, self).__init__()
        self.margin = margin
        self.tradeoff = tradeoff
        self.temperature = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.boost = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, logits, targets):
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + pow(10, -7)
        logits = torch.div(logits, norms + torch.relu(self.boost))
        logits = logits / torch.clamp(self.temperature, min=1)

        batch_size = logits.size()[0]
        true_class_logits = logits[torch.arange(batch_size), targets].unsqueeze(1)

        margins = self.margin + logits - true_class_logits
        margins[torch.arange(batch_size), targets] = 0

        return (
            torch.clamp(margins, min=0).sum(dim=1).mean() - self.tradeoff * logits.std()
        )
```

### Matlab

```matlab
classdef MarginEnhancedLogitNormLoss < handle
    properties
        temperature
        margin
    end

    methods
        function obj = MarginEnhancedLogitNormLoss(initial_temperature, initial_margin)
            if nargin > 0
                obj.temperature = initial_temperature;
                obj.margin = initial_margin;
            else
                obj.temperature = 1.0;
                obj.margin = 0.1;
            end
        end

        function loss = forward(obj, x, target)
            norms = sqrt(sum(x.^2, 2)) + 1e-7;
            logit_norm = (x - obj.margin) ./ norms / obj.temperature;
            loss = obj.cross_entropy(logit_norm, target);
        end

        function loss = cross_entropy(~, logit_norm, target)
            logits = exp(logit_norm);
            softmax_output = logits ./ sum(logits, 2);
            loss = -mean(log(target) .* softmax_output, 2);
        end
    end
end
```

## Dataset

The dataset used for this research training was the portraits dataset provided by [Dr. Nanni](https://scholar.google.it/citations?user=5NSGzcQAAAAJ&hl=en) to make sure the comparison between already implemented approaches and the new ideas is solid since he could instantly compare newly achieved performance metrics with the prior.

<div align="center">
    <img src="art/distribution.png" width="400" />
</div>

The dataset consists of a total of 927 images, meticulously categorized into six distinct groups. The figure provides a visual representation of the distribution of images across these categories, offering insights into the dataset's composition.

### Sample Images

<div align="center">
  <img src="art/dataset-image-1.jpg" width="100" />
  <img src="art/dataset-image-2.jpg" width="100" />
  <img src="art/dataset-image-3.jpg" width="100" />
</div>

<div align="center">
  <img src="art/dataset-image-4.jpg" width="100" />
  <img src="art/dataset-image-5.jpg" width="100" />
  <img src="art/dataset-image-6.jpg" width="100" />
</div>

<div align="center">
  <img src="art/dataset-image-7.jpg" width="100" />
  <img src="art/dataset-image-8.jpg" width="100" />
  <img src="art/dataset-image-9.jpg" width="100" />
</div>

## References

```bibtex
@article{wei2022logitnorm,
    title={Mitigating Neural Network Overconfidence with Logit Normalization},
    author={Wei, Hongxin and Xie, Renchunzi and Cheng, Hao and Feng, Lei and An, Bo and Li, Yixuan},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2022}
}
```
