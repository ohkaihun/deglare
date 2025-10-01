import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNormalizer(nn.Module):
    """输入预处理：四元数单位化 + 平移向量归一化"""

    def __init__(self, trans_scale=10.0):
        super().__init__()
        self.trans_scale = trans_scale  # 根据场景调整（如平移量级）

    def forward(self, R, T):
        # R: [B,4], T: [B,3]
        R_norm = R / (torch.norm(R, dim=1, keepdim=True) + 1e-6 ) # 单位四元数
        T_norm = T/self.trans_scale  # 平移归一化
        return R_norm, T_norm


class PositionEncoder(nn.Module):
    """优化版位置编码器：残差连接 + 注意力增强"""

    def __init__(self, hidden_dim=128):
        super().__init__()
        # 旋转编码分支（加深网络）
        self.rot_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 平移编码分支（加深网络）
        self.trans_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 交叉注意力（优化维度处理）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=False  # 原始维度顺序适配
        )

        # 残差融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, R, T):
        # 分叉编码
        rot_feat = self.rot_encoder(R)  # [B, D]
        trans_feat = self.trans_encoder(T)  # [B, D]

        # 交叉注意力（调整维度顺序）
        # 输入需要 [SeqLen, Batch, Dim]
        rot_feat_ = rot_feat.unsqueeze(0)  # [1, B, D]
        trans_feat_ = trans_feat.unsqueeze(0)  # [1, B, D]
        attn_out, _ = self.cross_attn(
            query=rot_feat_,
            key=trans_feat_,
            value=trans_feat_
        )  # [1, B, D]
        attn_out = attn_out.squeeze(0)  # [B, D]

        # 残差连接 + 拼接
        combined = torch.cat([rot_feat + attn_out, trans_feat], dim=1)  # [B, 2D]
        return self.fusion(combined)


class ParameterPredictor(nn.Module):
    """增强版参数预测：深度网络 + 物理约束强化"""

    def __init__(self, width, height, hidden_dim, output_dim=4):
        super().__init__()
        self.width = width
        self.height = height

        # 深层预测网络
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化增强收敛速度"""
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, mean=0.1, std=0.02)

    def forward(self, x):
        output = self.predictor(x)
        amplitude, sigma, alpha, offset = output.split(1, dim=1)

        # 物理约束优化
        M = 3+torch.sigmoid(amplitude) * 5  # 振幅 (0, 50)
        S = 1 + F.softplus(sigma)  # sigma >=0.1
        A = 0.7+ 0.5 * torch.sigmoid(alpha)  # alpha (1, 3)
        Offset = torch.sigmoid(offset) * 0.01  # 基底亮度偏移 (0, 0.1)

        return M, S, A, Offset


class HyperlaplacePredictor(nn.Module):
    """完整模型：集成预处理 + 编码 + 预测"""

    def __init__(self, width, height, hidden_dim=128, device='cuda'):
        super().__init__()
        self.normalizer = PoseNormalizer(trans_scale=10.0)
        self.encoder = PositionEncoder(hidden_dim=hidden_dim)
        self.predictor = ParameterPredictor(width, height, hidden_dim)
        self.to(device)

    def forward(self, R_raw, T_raw):
        # 输入预处理
        R, T = self.normalizer(R_raw, T_raw)

        # 特征编码
        features = self.encoder(R, T)

        # 参数预测
        return self.predictor(features)

class HyperLaplacianNet(nn.Module):
    def __init__(self, rt_dim=12,output_dim=1,ch=3,device='cuda'):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(rt_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Linear(64, 6)
        # 输出 alpha, A, sigma, offset
        )

        # 分别为三个通道的 alpha 和 sigma 创建输出层
        self.alpha_output = nn.Linear(256, output_dim*ch)  # 输出 3 个 alpha
        self.sigma_output = nn.Linear(256, output_dim*ch)  # 输出 3 个 sigma
        self.to(device)
    def forward(self, R,T):
        rt_matrix = torch.cat([R , T], dim=1)  # [B, 2D]
        # params = self.main(rt_matrix)
        features = self.main(rt_matrix)  # 经过前几层

        # 使用独立的输出层
        alpha = 0.2 + 0.7 * torch.sigmoid(self.alpha_output(features))  # (B, 3)
        sigma = 8* torch.sigmoid(self.sigma_output(features))            # (B, 3)

        return  sigma,alpha

