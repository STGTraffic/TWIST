from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

   
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, 128))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):

        day_emb = x[..., 1]  
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]  
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]  
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]  
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
        
class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=False, device=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.trend_aware=False
        
        if self.trend_aware:
            kernel_size = 3
            self.qk_conv = nn.Conv1d(dim, dim * 2, kernel_size=kernel_size, padding=kernel_size // 2)
            self.v_fc = nn.Linear(dim, dim)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.mask = torch.tril(torch.ones(window_size, window_size)).to(
            device)  # mask for causality

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)  # create local windows
        B, T, C = x.shape
        if self.trend_aware:
            v = self.v_fc(x).reshape(B, T, self.num_heads, 
                                     C // self.num_heads).permute(0, 2, 1, 3)
            #print(v.shape)
            x = x.permute(0, 2, 1)  #  [B, C, T]
            qk = self.qk_conv(x)  
            qk = qk.permute(0, 2, 1)  # [B, T, C*2]
            qk = qk.reshape(B, T, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qk[0], qk[1]
        else:
            qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        # merge key padding and attention masks
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, T, T]

        if self.causal:
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        if self.window_size > 0:  # reshape to the original size
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class MTWSA(nn.Module):
    def __init__(self,
                 dim = 128, 
                 depth = 4, 
                 heads = 2,  
                 window_size = 12,  
                 mlp_dim = 64,  
                 num_time = 12,  
                 dropout=0.,  
                 device='cuda:0'):  
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([
            TemporalAttention(dim=dim, 
                              heads=heads, 
                              window_size=window_size, 
                              dropout=dropout, 
                              device=device)
                for _ in range(depth)  
            ])
        self.dim = dim
        self.depth = depth
        self.heads = heads 
        self.window_size = window_size
        self.mlp_dim = mlp_dim
        self.num_time = num_time

    def forward(self, x):
        b, c, n, t = x.shape
        res = x
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)  # [b*n, t, c]
        x = x + self.pos_embedding  # [b*n, t, c]
        for attn in self.layers:
            x = attn(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        x = x[..., -1].unsqueeze(-1) + res[..., -1].unsqueeze(-1)
        return x


class Encoder(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head 
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.attention = SpatialAttention(factor=5, scale=None, attention_dropout=0.1, num_nodes=self.num_nodes)
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        self.q = Conv(d_model)
        self.k = Conv(d_model)
        self.v = Conv(d_model)
        

    def forward(self, input, adj_list=None):
        # 64 64 170 12
        #print('input', input.shape)        #input torch.Size([64, 256, 170, 1])
        
        Q, K, V = self.q(input), self.k(input), self.v(input)
        Q = Q.permute(0, 2, 3, 1)  #torch.Size([64, 170, 1, 256])
        K = K.permute(0, 2, 3, 1)  
        V = V.permute(0, 2, 3, 1)  
        B, N, t, d = Q.shape
        
        Q = Q.view(B, N, t, 2, 128).transpose(1, 3)#Q torch.Size([64, 2, 1, 170, 128])  
        K = K.view(B, N, t, 2, 128).transpose(1, 3)
        V = V.view(B, N, t, 2, 128).transpose(1, 3)
        x, weight, bias = self.attention(Q, K, V)

        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x
        

class SpatialAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, num_nodes=None):
        super(SpatialAttention, self).__init__()

        self.factor = factor
        self.scale = scale
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(attention_dropout)
        self.weight = nn.Parameter(torch.ones(256, self.num_nodes, 1))
        self.bias = nn.Parameter(torch.zeros(256, self.num_nodes, 1))
        self.linear = Conv(256)
        self.lambda_weight = nn.Parameter(torch.tensor(0.5)) 
    
    def _QK(self, Q, K, sample_k, n_top):
        B, H, T, N, D = Q.shape

        index_sample = torch.randint(0, N, (sample_k,), device=Q.device)  # 随机采样 K
        K_sample = K[:, :, :, index_sample, :]  # 只取部分 K 进行计算

        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))  # (B, H, T, N, sample_k)

        S = Q_K_sample
        p = S / (S.sum(dim=-1, keepdim=True) + 1e-10)  # Normalize
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # Compute entropy

        entropy_topk, index = entropy.topk(n_top, dim=-1, largest=False, sorted=False)

        Q_reduce = Q.gather(dim=3, index=index.unsqueeze(-1).expand(-1, -1, -1, -1, D))
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # (B, H, T, n_top, N) #calculate att between significant Q' and Keys

        return Q_K, index
    
    def _get_initial_context(self, V, N):
        B, H, T, N, D = V.shape
        V_sum = V.mean(dim=-2)  # Average V
        context = V_sum.unsqueeze(-2).expand(B, H, T, N, V_sum.shape[-1]).clone()

        return context

    def _update_context(self, context_in, V, scores, index, N, Q=None, K=None):
        B, H, T, _, D = V.shape
    
        attn = torch.softmax(scores, dim=-1)#torch.Size([64, 2, 1, 30, 250])

        context_activate = torch.matmul(attn, V).type_as(context_in)                      # update activate nodes
        
        batch_idx = torch.arange(B, device=V.device)[:, None, None, None]
        head_idx = torch.arange(H, device=V.device)[None, :, None, None]
        time_idx = torch.arange(T, device=V.device)[None, None, :, None]

        #find most relevent nodes for passive nodes according to sim_weights
        sim_weights = attn
        context_global = torch.matmul(sim_weights.transpose(-2, -1), context_activate)     # update all nodes using active nodes
        #print('context_passive', context_passive.shape)#context_passive torch.Size([64, 2, 1, 170, 128])
        context_global = context_global.scatter_(                                          # covering active nodes
            dim=-2,
            index=index.unsqueeze(-1).expand(-1, -1, -1, -1, D),
            src=context_activate)

        return context_global


    def forward(self, queries, keys, values):
        B, H, T, N, D = queries.shape
        
        U_part = self.factor * np.ceil(np.log(N)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(N)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < N else N
        u = u if u < N else N

        scores_top, index = self._QK(queries, keys, sample_k=U_part, n_top=u)
        # Add scale factor
        scale = 1. / math.sqrt(D)
        scores_top = scores_top * scale
        # Get the context
        context = self._get_initial_context(values, N)

        context = self._update_context(context, values, scores_top, index, N, queries, keys)

        context = context.permute(0, 3, 2, 1, 4).contiguous()
        context = context.reshape(B, -1, N, T)
        x = self.linear(context)
        if self.num_nodes not in [170, 358,5]:
            x = x * self.weight + self.bias + x
        return x, self.weight, self.bias

class TWIST(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=883,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1
        self.blocks = 4

        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358  or num_nodes == 883 or num_nodes == 1918:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes >200:
            time = 96

        self.Temb = TemporalEmbedding(time, channels)

        

        self.start_conv = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))  

        self.network_channel = channels * 2


        self.TW_attetion = MTWSA(dim = self.node_dim, depth = 2, heads = 1, 
                                  window_size = 12, mlp_dim= 64, num_time = self.input_len,  dropout = 0., device= self.device)

        self.SpatialBlock = Encoder(
            device,
            d_model=self.network_channel,
            head=self.head,
            num_nodes=num_nodes,
            seq_length=1,
            dropout=dropout,
        )

        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )

        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )


    def param_num_layer(self):
        total_params = 0
        for name, param in self.named_parameters():
            param_count = param.numel()  # 获取每个参数张量的元素数量
            total_params += param_count
            print(f"Layer: {name}, Parameters: {param_count}")
        return total_params

    def forward(self, history_data):
        #print('history_data', history_data.shape)                #history_data torch.Size([64, 3, 307, 12])
        input_data = history_data
        history_data = history_data.permute(0, 3, 2, 1)
        input_data = self.start_conv(input_data)                   

        input_data = self.TW_attetion(input_data)
        tem_emb = self.Temb(history_data)
        data_st = torch.cat([input_data] + [tem_emb], dim=1)

        data_st = self.SpatialBlock(data_st) + self.fc_st(data_st)

        prediction = self.regression_layer(data_st)

        return prediction
