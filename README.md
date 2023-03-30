## SHIKE
**Long-Tailed Visual Recognition via Self-Heterogeneous Integration with Knowledge Excavation**
**Authors**: Yan Jin, [Mengke Li](https://github.com/Keke921), [Yang Lu*](https://jasonyanglu.github.io), Yiu-ming Cheung, Hanzi Wang

This is the repository of the CVPR 2023 paper: "Long-Tailed Visual Recognition via Self-Heterogeneous Integration with Knowledge Excavation." We find that deep neural networks have different preferences towards the long-tailed distribution according to the depth. SHIKE is designed as a Mixture of Experts (MoE) method, which fuses  features of different depths and enables transfer among experts,  boosting the performance effectively in long-tailed visual recognition. 


#### Requirements
```
python  3.7.7 or above
torch   1.11.0 or above
```

### Implementation for all datasets is still under reoganizing...
stay tuned for it～

### Acknowledgement
Data augmentation in SHIKE mainly follows [BalancedMetaSoftmax](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification) and [PaCo](https://github.com/dvlab-research/Parametric-Contrastive-Learning).
