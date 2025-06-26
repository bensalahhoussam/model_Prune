## Model Pruning with Taylor Ranking
This repository provides an implementation of structured model pruning using the Taylor expansion based ranking criterion. The technique evaluates the importance of model channels/filters by estimating the change in loss when each channel is removed, using a first-order Taylor expansion.

🚀 Key Features
Taylor Ranking for channel importance estimation

Support for structured pruning (e.g., filters, channels, layers)

Designed for CNN-based architectures (e.g., ResNet, ,Unet, VGG)

Compatible with PyTorch

Option to prune multiple layers and fine-tune the network afterward

📘 What is Taylor Pruning?
Taylor Pruning leverages the first-order Taylor expansion of the loss function to estimate how much the loss would increase if a particular parameter (e.g., filter or channel) is removed. The magnitude of this approximation serves as an importance score for pruning.

Mathematically:

ΔL ≈ ∂L/∂w × w

The smaller the score, the less impact the weight/channel has on the loss — making it a candidate for pruning.

## Run 
This performs ranking, removal, finetuning and evaluation in one pruning iteration.  
```python prune.py --load YOUR_MODEL.pth --channel_txt YOUR_CHANNELS.txt```


| Iteration | Ranking Iterations | Pruned Channels | Validation DICE | File Size (MB) |
|-----------|--------------------|------------------|------------------|----------------|
| 0         | N/A                | N/A              | 0.975            | 52.4           |
| 1         | 500                | 300              | 0.968            | 44.4           |
| 2         | 500                | 300              | 0.960            | 38.9           |
| 3         | 500                | 300              | 0.957            | 33.2           |
| 4         | 500                | 300              | 0.951            | 27.2           |
