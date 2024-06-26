Our code-base has been significantly influenced by the repository from the paper ["Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale"](https://github.com/amazon-science/llm-interpret)

Instructions for running our code is in the parent directory.

We re-implement a _Deja-Vu style_ predictor in this code-base, along with our ShadowLLM predictor. We simplify the DejaVu implementation by having per-layer predictors instead of look-ahead predictors. This variant should ideally have better accuracy characteristics as there are more sparsity predictors in our test.
