# unlearnable
For AAAI 2026 anonymous submission "Towards Provably Unlearnable Examples via Bayes Error Optimization"


---

## 📜 Overview

This repository implements the methods and experiments described in the manuscript, allowing you to generate unlearnable examples using our novel approach.
---

To get started, first clone the repo and install the necessary dependencies:

```bash
git clone https://huggingface.co/your-username/unlearnable
cd unlearnable
pip install -r requirements.txt
```

1. Once the environment is set up, run the following command to generate unlearnable examples using the main code:

```
python perturb.py
```

This will generate unlearnable examples based on the method described in the paper.

2. Datasets and Comparisons
The generated unlearnable examples are provided alongside examples from existing methods, available in a well-maintained Hugging Face repository, which can be found [here](https://huggingface.co/datasets/soumission-anonyme/unlearnable/tree/main).

You can also find the original CIFAR-10 train-test split in this Hugging Face repo. For those interested in Tiny ImageNet, a corresponding dataset is available [here](https://huggingface.co/datasets/zh-plus/tiny-imagenet).

3. Training on Unlearnable Examples
Once unlearnable examples are generated, you could train on them to evaluate performance degradation.



