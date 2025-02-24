# T-Graphormer

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/t-graphormer-using-transformers-for/traffic-prediction-on-pems-bay)](https://paperswithcode.com/sota/traffic-prediction-on-pems-bay?p=t-graphormer-using-transformers-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/t-graphormer-using-transformers-for/traffic-prediction-on-metr-la)](https://paperswithcode.com/sota/traffic-prediction-on-metr-la?p=t-graphormer-using-transformers-for)

Multivariate time series data is ubiquitous, and forecasting it has important applications in many domains. However, its complex spatial dependencies and non-linear temporal dynamics can be challenging for traditional techniques. Existing methods tackle these challenges by learning the two dimensions separately. Here, we introduce Temporal Graphormer (T-Graphormer), a Transformer-based approach capable of modelling spatiotemporal correlations simultaneously. By incorporating temporal dynamics in the Graphormer architecture, each node attends to all other nodes within the graph sequence. Our design enables the model to capture rich spatiotemporal patterns with minimal reliance on predefined spacetime inductive biases. We validate the effectiveness of T-Graphormer on real-world traffic prediction benchmark datasets. Compared to state-of-the-art methods, T-Graphormer reduces root mean squared error (RMSE) and mean absolute percentage error (MAPE) by up to 10%.

For full paper, see [https://www.arxiv.org/abs/2501.13274](https://www.arxiv.org/abs/2501.13274)
