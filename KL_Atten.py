import torch.nn.functional as F
from torch import nn
import torch

class KLattention(nn.Module):
    def __init__(self, window_size, stride, input_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.out = nn.Linear(input_size, input_size)

    def kl_divergence(self, p, q):
        return torch.sum(p * (torch.log(p) - torch.log(q)))

    def MultiheadAttention(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, -1)
        out = self.out(out)
        return out

    def recursive_multihead_attention(self, x, depth=0):
        if depth == 1:
            return x
        else:
            output = self.MultiheadAttention(x)
            return self.recursive_multihead_attention(output, depth + 1)

    def forward(self, x, klcls):
        segments = []

        for i in range(0, x.size(1) - self.window_size + 1, self.stride):
            segment = x[:, i:i + self.window_size]
            segments.append(segment)
        kl_divergences = []
        prev_segment_distribution = None
        for segment in segments:
            segment_distribution = F.softmax(segment, dim=1)
            if prev_segment_distribution is not None:
                kl_divergence_value = self.kl_divergence(segment_distribution, prev_segment_distribution)
                kl_divergences.append(kl_divergence_value.item())
            prev_segment_distribution = segment_distribution

        top_k_indices = sorted(range(len(kl_divergences)), key=lambda i: kl_divergences[i])[:3]
        top_k_segments = [torch.cat([klcls, segments[i], segments[i + 1]], dim=1) for i in top_k_indices]

        output_list = [self.recursive_multihead_attention(list) for list in top_k_segments]
        tensor_split = [(tensor[:, :1, :], tensor[:, 1:, :]) for tensor in output_list]

        klcls_1 = torch.cat([item[0] for item in tensor_split], dim=1)  # (32, n, 400)
        klTrans = torch.cat([item[1] for item in tensor_split], dim=1)

        return klTrans, klcls_1