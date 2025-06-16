import torch
scores = torch.tensor([3.0, 1.0, 2.0])
exp_scores = torch.exp(scores)
# → [e^3, e^1, e^2] ≈ [20.1, 2.7, 7.4]
sum_exp = torch.sum(exp_scores)
# → 20.1 + 2.7 + 7.4 = 30.2
softmax = exp_scores / sum_exp
# → [20.1/30.2, 2.7/30.2, 7.4/30.2] ≈ [0.665, 0.089, 0.245]
result = torch.tensor([0.665, 0.089, 0.245])

print("scores:",scores)
print("exp_scores:",exp_scores)
print("sum_exp:",sum_exp)
print("softmax:",softmax)
print("result:",result)
