"""
选手可参考以下流程，使用提供的 RQ-VAE 框架代码将多模态emb数据转换为Semantic Id:
1. 使用 MmEmbDataset 读取不同特征 ID 的多模态emb数据.
2. 训练 RQ-VAE 模型, 训练完成后将数据转换为Semantic Id.
3. 参照 Item Sparse 特征格式处理Semantic Id，作为新特征加入Baseline模型训练.
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os
from pathlib import Path
from dataset import load_mm_emb


class MmEmbDataset(torch.utils.data.Dataset):
    """
    Build Dataset for RQ-VAE Training

    Args:
        data_dir = os.environ.get('TRAIN_DATA_PATH')
        feature_id = MM emb ID
    """

    def __init__(self, data_dir, feature_id):
        """
        初始化多模态嵌入数据集

        Args:
            data_dir (str): 数据目录路径
            feature_id (str): 多模态嵌入特征ID，如'81', '82'等
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.mm_emb_id = [feature_id]
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_id)

        self.mm_emb = self.mm_emb_dict[self.mm_emb_id[0]]
        self.tid_list, self.emb_list = list(self.mm_emb.keys()), list(self.mm_emb.values())
        self.emb_list = [torch.tensor(emb, dtype=torch.float32) for emb in self.emb_list]

        assert len(self.tid_list) == len(self.emb_list)
        self.item_cnt = len(self.tid_list)

    def __getitem__(self, index):
        """
        获取指定索引的数据项

        Args:
            index (int): 数据项索引

        Returns:
            tuple: (tid, emb) - 物品ID和对应的嵌入向量
        """
        tid = torch.tensor(self.tid_list[index], dtype=torch.long)
        emb = self.emb_list[index]
        return tid, emb

    def __len__(self):
        """
        返回数据集大小

        Returns:
            int: 数据集中物品的数量
        """
        return self.item_cnt

    @staticmethod
    def collate_fn(batch):
        """
        批处理函数，将多个数据项组合成批次

        Args:
            batch (list): 包含多个(tid, emb)元组的列表

        Returns:
            tuple: (tid_batch, emb_batch) - 批次化的物品ID和嵌入向量
        """
        tid, emb = zip(*batch)

        tid_batch, emb_batch = torch.stack(tid, dim=0), torch.stack(emb, dim=0)
        return tid_batch, emb_batch


## Kmeans
def kmeans(data, n_clusters, kmeans_iters):
    """
    使用sklearn的KMeans算法对数据进行聚类

    Args:
        data (torch.Tensor): 输入数据，形状为[N, D]
        n_clusters (int): 聚类中心数量
        kmeans_iters (int): 最大迭代次数

    Returns:
        tuple: (cluster_centers, labels) - 聚类中心和标签
            - cluster_centers (torch.Tensor): 聚类中心，形状为[n_clusters, D]
            - labels (torch.Tensor): 每个数据点的聚类标签，形状为[N]

    Note:
        auto init: n_init = 10 if n_clusters <= 10 else 1
    """
    km = KMeans(n_clusters=n_clusters, max_iter=kmeans_iters, n_init="auto")

    # sklearn only support cpu
    data_cpu = data.detach().cpu()
    np_data = data_cpu.numpy()
    km.fit(np_data)
    return torch.tensor(km.cluster_centers_), torch.tensor(km.labels_)


## Balanced Kmeans
class BalancedKmeans(torch.nn.Module):
    """
    平衡K-means聚类算法，确保每个聚类中心分配到相近数量的数据点

    Args:
        num_clusters (int): 聚类中心数量
        kmeans_iters (int): 最大迭代次数
        tolerance (float): 收敛容忍度
        device (str): 计算设备 ('cpu' 或 'cuda')
    """

    def __init__(self, num_clusters: int, kmeans_iters: int, tolerance: float, device: str):
        """
        初始化平衡K-means聚类器

        Args:
            num_clusters (int): 聚类中心数量
            kmeans_iters (int): 最大迭代次数
            tolerance (float): 收敛容忍度，当聚类中心变化小于此值时停止迭代
            device (str): 计算设备 ('cpu' 或 'cuda')
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_iters = kmeans_iters
        self.tolerance = tolerance
        self.device = device
        self._codebook = None

    def _compute_distances(self, data):
        """
        计算数据点到聚类中心的距离

        Args:
            data (torch.Tensor): 输入数据，形状为[N, D]

        Returns:
            torch.Tensor: 距离矩阵，形状为[N, num_clusters]
        """
        return torch.cdist(data, self._codebook)

    def _assign_clusters(self, dist):
        """
        平衡地分配数据点到聚类中心，确保每个聚类的大小相近

        Args:
            dist (torch.Tensor): 距离矩阵，形状为[N, num_clusters]

        Returns:
            torch.Tensor: 每个数据点的聚类标签，形状为[N]
        """
        samples_cnt = dist.shape[0]
        samples_labels = torch.zeros(samples_cnt, dtype=torch.long, device=self.device)
        clusters_cnt = torch.zeros(self.num_clusters, dtype=torch.long, device=self.device)

        sorted_indices = torch.argsort(dist, dim=-1)
        for i in range(samples_cnt):
            for j in range(self.num_clusters):
                cluster_idx = sorted_indices[i, j]
                if clusters_cnt[cluster_idx] < samples_cnt // self.num_clusters:
                    samples_labels[i] = cluster_idx
                    clusters_cnt[cluster_idx] += 1
                    break

        return samples_labels

    def _update_codebook(self, data, samples_labels):
        """
        根据分配的聚类标签更新聚类中心

        Args:
            data (torch.Tensor): 输入数据，形状为[N, D]
            samples_labels (torch.Tensor): 每个数据点的聚类标签，形状为[N]

        Returns:
            torch.Tensor: 更新后的聚类中心，形状为[num_clusters, D]
        """
        _new_codebook = []
        for i in range(self.num_clusters):
            cluster_data = data[samples_labels == i]
            if len(cluster_data) > 0:
                _new_codebook.append(cluster_data.mean(dim=0))
            else:
                _new_codebook.append(self._codebook[i])
        return torch.stack(_new_codebook)

    def fit(self, data):
        """
        训练平衡K-means聚类器

        Args:
            data (torch.Tensor): 输入数据，形状为[N, D]

        Returns:
            tuple: (codebook, labels) - 最终的聚类中心和标签
                - codebook (torch.Tensor): 聚类中心，形状为[num_clusters, D]
                - labels (torch.Tensor): 每个数据点的聚类标签，形状为[N]
        """
        num_emb, codebook_emb_dim = data.shape
        data = data.to(self.device)

        # initialize codebook
        indices = torch.randperm(num_emb)[: self.num_clusters]
        self._codebook = data[indices].clone()

        for _ in range(self.kmeans_iters):
            dist = self._compute_distances(data)
            samples_labels = self._assign_clusters(dist)
            _new_codebook = self._update_codebook(data, samples_labels)
            if torch.norm(_new_codebook - self._codebook) < self.tolerance:
                break

            self._codebook = _new_codebook

        return self._codebook, samples_labels

    def predict(self, data):
        """
        对新数据进行聚类预测

        Args:
            data (torch.Tensor): 输入数据，形状为[N, D]

        Returns:
            torch.Tensor: 每个数据点的聚类标签，形状为[N]
        """
        data = data.to(self.device)
        dist = self._compute_distances(data)
        samples_labels = self._assign_clusters(dist)
        return samples_labels


## Base RQVAE
class RQEncoder(torch.nn.Module):
    """
    RQ-VAE编码器，将输入数据编码为潜在表示

    Args:
        input_dim (int): 输入数据的维度
        hidden_channels (list): 隐藏层维度列表，如[512, 256]
        latent_dim (int): 潜在空间的维度
    """

    def __init__(self, input_dim: int, hidden_channels: list, latent_dim: int):
        """
        初始化RQ-VAE编码器

        Args:
            input_dim (int): 输入数据的维度
            hidden_channels (list): 隐藏层维度列表，按顺序构建网络层
            latent_dim (int): 潜在空间的维度
        """
        super().__init__()

        self.stages = torch.nn.ModuleList()
        in_dim = input_dim

        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim

        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, latent_dim), torch.nn.ReLU()))

    def forward(self, x):
        """
        前向传播，将输入编码为潜在表示

        Args:
            x (torch.Tensor): 输入数据，形状为[batch_size, input_dim]

        Returns:
            torch.Tensor: 编码后的潜在表示，形状为[batch_size, latent_dim]
        """
        for stage in self.stages:
            x = stage(x)
        return x


class RQDecoder(torch.nn.Module):
    """
    RQ-VAE解码器，将潜在表示解码为原始数据

    Args:
        latent_dim (int): 潜在空间的维度
        hidden_channels (list): 隐藏层维度列表，如[256, 512]
        output_dim (int): 输出数据的维度
    """

    def __init__(self, latent_dim: int, hidden_channels: list, output_dim: int):
        """
        初始化RQ-VAE解码器

        Args:
            latent_dim (int): 潜在空间的维度
            hidden_channels (list): 隐藏层维度列表，按顺序构建网络层
            output_dim (int): 输出数据的维度
        """
        super().__init__()

        self.stages = torch.nn.ModuleList()
        in_dim = latent_dim

        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim

        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, output_dim), torch.nn.ReLU()))

    def forward(self, x):
        """
        前向传播，将潜在表示解码为原始数据

        Args:
            x (torch.Tensor): 潜在表示，形状为[batch_size, latent_dim]

        Returns:
            torch.Tensor: 解码后的数据，形状为[batch_size, output_dim]
        """
        for stage in self.stages:
            x = stage(x)
        return x


## Generate semantic id
class VQEmbedding(torch.nn.Embedding):
    """
    向量量化嵌入层，用于生成语义ID

    继承自torch.nn.Embedding，通过聚类方法将连续向量映射到离散的语义ID

    Args:
        num_clusters (int): 聚类中心数量，即码本大小
        codebook_emb_dim (int): 码本嵌入维度
        kmeans_method (str): K-means方法，'kmeans'或'bkmeans'
        kmeans_iters (int): K-means最大迭代次数
        distances_method (str): 距离计算方法，'cosine'或'l2'
        device (str): 计算设备
    """

    def __init__(
        self,
        num_clusters,
        codebook_emb_dim: int,
        kmeans_method: str,
        kmeans_iters: int,
        distances_method: str,
        device: str,
    ):
        """
        初始化向量量化嵌入层

        Args:
            num_clusters (int): 聚类中心数量，即码本大小
            codebook_emb_dim (int): 码本嵌入维度
            kmeans_method (str): K-means方法，'kmeans'使用标准K-means，'bkmeans'使用平衡K-means
            kmeans_iters (int): K-means算法的最大迭代次数
            distances_method (str): 距离计算方法，'cosine'使用余弦距离，其他使用L2距离
            device (str): 计算设备 ('cpu' 或 'cuda')
        """
        super(VQEmbedding, self).__init__(num_clusters, codebook_emb_dim)

        self.num_clusters = num_clusters
        self.codebook_emb_dim = codebook_emb_dim
        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.device = device

    def _create_codebook(self, data):
        """
        创建码本，使用K-means聚类初始化聚类中心

        Args:
            data (torch.Tensor): 输入数据，形状为[N, D]
        """
        if self.kmeans_method == 'kmeans':
            _codebook, _ = kmeans(data, self.num_clusters, self.kmeans_iters)
        elif self.kmeans_method == 'bkmeans':
            BKmeans = BalancedKmeans(
                num_clusters=self.num_clusters, kmeans_iters=self.kmeans_iters, tolerance=1e-4, device=self.device
            )
            _codebook, _ = BKmeans.fit(data)
        else:
            _codebook = torch.randn(self.num_clusters, self.codebook_emb_dim)
        _codebook = _codebook.to(self.device)
        assert _codebook.shape == (self.num_clusters, self.codebook_emb_dim)
        self.codebook = torch.nn.Parameter(_codebook)

    @torch.no_grad()
    def _compute_distances(self, data):
        """
        计算数据点到码本中心的距离

        Args:
            data (torch.Tensor): 输入数据，形状为[N, D]

        Returns:
            torch.Tensor: 距离矩阵，形状为[N, num_clusters]
        """
        _codebook_t = self.codebook.t()
        assert _codebook_t.shape == (self.codebook_emb_dim, self.num_clusters)
        assert data.shape[-1] == self.codebook_emb_dim

        if self.distances_method == 'cosine':
            data_norm = F.normalize(data, p=2, dim=-1)
            _codebook_t_norm = F.normalize(_codebook_t, p=2, dim=0)
            distances = 1 - torch.mm(data_norm, _codebook_t_norm)
        # l2
        else:
            data_norm_sq = data.pow(2).sum(dim=-1, keepdim=True)
            _codebook_t_norm_sq = _codebook_t.pow(2).sum(dim=0, keepdim=True)
            distances = torch.addmm(data_norm_sq + _codebook_t_norm_sq, data, _codebook_t, beta=1.0, alpha=-2.0)
        return distances

    @torch.no_grad()
    def _create_semantic_id(self, data):
        """
        根据距离创建语义ID，选择最近的聚类中心

        Args:
            data (torch.Tensor): 输入数据，形状为[N, D]

        Returns:
            torch.Tensor: 语义ID，形状为[N]
        """
        distances = self._compute_distances(data)
        _semantic_id = torch.argmin(distances, dim=-1)
        return _semantic_id

    def _update_emb(self, _semantic_id):
        """
        根据语义ID更新嵌入向量

        Args:
            _semantic_id (torch.Tensor): 语义ID，形状为[N]

        Returns:
            torch.Tensor: 更新后的嵌入向量，形状为[N, codebook_emb_dim]
        """
        update_emb = super().forward(_semantic_id)
        return update_emb

    def forward(self, data):
        """
        前向传播，将连续向量量化为离散的语义ID和对应的嵌入向量

        Args:
            data (torch.Tensor): 输入数据，形状为[N, D]

        Returns:
            tuple: (update_emb, _semantic_id)
                - update_emb (torch.Tensor): 量化后的嵌入向量，形状为[N, codebook_emb_dim]
                - _semantic_id (torch.Tensor): 语义ID，形状为[N]
        """
        self._create_codebook(data)
        _semantic_id = self._create_semantic_id(data)
        update_emb = self._update_emb(_semantic_id)

        return update_emb, _semantic_id


## Residual Quantizer
class RQ(torch.nn.Module):
    """
    残差量化器，实现多层级的向量量化

    通过多个码本进行残差量化，每一层量化前一层的残差，从而实现更精细的量化效果

    Args:
        num_codebooks (int): 码本数量，即量化层数
        codebook_size (list): 每个码本的大小列表
        codebook_emb_dim (int): 码本嵌入维度
        shared_codebook (bool): 是否共享码本
        kmeans_method (str): K-means初始化方法
        kmeans_iters (int): K-means迭代次数
        distances_method (str): 距离计算方法
        loss_beta (float): RQ-VAE损失的权重系数
        device (str): 计算设备
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: list,
        codebook_emb_dim,
        shared_codebook: bool,
        kmeans_method,
        kmeans_iters,
        distances_method,
        loss_beta: float,
        device: str,
    ):
        """
        初始化残差量化器

        Args:
            num_codebooks (int): 码本数量，即量化层数
            codebook_size (list): 每个码本的大小列表，长度应等于num_codebooks
            codebook_emb_dim (int): 码本嵌入维度
            shared_codebook (bool): 是否在所有层之间共享码本
            kmeans_method (str): K-means初始化方法，'kmeans'或'bkmeans'
            kmeans_iters (int): K-means算法的最大迭代次数
            distances_method (str): 距离计算方法，'cosine'或'l2'
            loss_beta (float): RQ-VAE损失中的权重系数
            device (str): 计算设备 ('cpu' 或 'cuda')
        """
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        assert len(self.codebook_size) == self.num_codebooks
        self.codebook_emb_dim = codebook_emb_dim
        self.shared_codebook = shared_codebook

        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.loss_beta = loss_beta
        self.device = device

        if self.shared_codebook:
            self.vqmodules = torch.nn.ModuleList(
                [
                    VQEmbedding(
                        self.codebook_size[0],
                        self.codebook_emb_dim,
                        self.kmeans_method,
                        self.kmeans_iters,
                        self.distances_method,
                        self.device,
                    )
                    for _ in range(self.num_codebooks)
                ]
            )

        else:
            self.vqmodules = torch.nn.ModuleList(
                [
                    VQEmbedding(
                        self.codebook_size[idx],
                        self.codebook_emb_dim,
                        self.kmeans_method,
                        self.kmeans_iters,
                        self.distances_method,
                        self.device,
                    )
                    for idx in range(self.num_codebooks)
                ]
            )

    def quantize(self, data):
        """
        执行残差量化过程

        残差量化的核心思想是逐层量化残差：
        - 第i层量化：input[i] (即 res[i-1]) = VQ[i] + res[i]
        - vq_emb_list: [vq1, vq1+vq2, ...] 累积的量化嵌入
        - res_emb_list: [res1, res2, ...] 每层的残差
        - semantic_id_list: [vq1_sid, vq2_sid, ...] 每层的语义ID

        Args:
            data (torch.Tensor): 输入数据，形状为[batch_size, codebook_emb_dim]

        Returns:
            tuple: (vq_emb_list, res_emb_list, semantic_id_list)
                - vq_emb_list (list): 累积量化嵌入列表，每个元素形状为[batch_size, codebook_emb_dim]
                - res_emb_list (list): 残差嵌入列表，每个元素形状为[batch_size, codebook_emb_dim]
                - semantic_id_list (torch.Tensor): 语义ID，形状为[batch_size, num_codebooks]
        """
        res_emb = data.detach().clone()

        vq_emb_list, res_emb_list = [], []
        semantic_id_list = []
        vq_emb_aggre = torch.zeros_like(data)

        for i in range(self.num_codebooks):
            vq_emb, _semantic_id = self.vqmodules[i](res_emb)

            res_emb -= vq_emb
            vq_emb_aggre += vq_emb

            res_emb_list.append(res_emb)
            vq_emb_list.append(vq_emb_aggre)
            semantic_id_list.append(_semantic_id.unsqueeze(dim=-1))

        semantic_id_list = torch.cat(semantic_id_list, dim=-1)
        return vq_emb_list, res_emb_list, semantic_id_list

    def _rqvae_loss(self, vq_emb_list, res_emb_list):
        """
        计算RQ-VAE损失函数

        Args:
            vq_emb_list (list): 累积量化嵌入列表
            res_emb_list (list): 残差嵌入列表

        Returns:
            torch.Tensor: RQ-VAE损失值
        """
        rqvae_loss_list = []
        for idx, quant in enumerate(vq_emb_list):
            # stop gradient
            loss1 = (res_emb_list[idx].detach() - quant).pow(2.0).mean()
            loss2 = (res_emb_list[idx] - quant.detach()).pow(2.0).mean()
            partial_loss = loss1 + self.loss_beta * loss2
            rqvae_loss_list.append(partial_loss)

        rqvae_loss = torch.sum(torch.stack(rqvae_loss_list))
        return rqvae_loss

    def forward(self, data):
        """
        前向传播，执行残差量化并计算损失

        Args:
            data (torch.Tensor): 输入数据，形状为[batch_size, codebook_emb_dim]

        Returns:
            tuple: (vq_emb_list, semantic_id_list, rqvae_loss)
                - vq_emb_list (list): 累积量化嵌入列表
                - semantic_id_list (torch.Tensor): 语义ID，形状为[batch_size, num_codebooks]
                - rqvae_loss (torch.Tensor): RQ-VAE损失值
        """
        vq_emb_list, res_emb_list, semantic_id_list = self.quantize(data)
        rqvae_loss = self._rqvae_loss(vq_emb_list, res_emb_list)

        return vq_emb_list, semantic_id_list, rqvae_loss


class RQVAE(torch.nn.Module):
    """
    残差量化变分自编码器(RQ-VAE)，用于将多模态嵌入转换为语义ID

    结合了编码器、解码器和残差量化器，实现端到端的向量量化学习

    Args:
        input_dim (int): 输入数据维度
        hidden_channels (list): 隐藏层维度列表
        latent_dim (int): 潜在空间维度
        num_codebooks (int): 码本数量
        codebook_size (list): 每个码本的大小
        shared_codebook (bool): 是否共享码本
        kmeans_method (str): K-means初始化方法
        kmeans_iters (int): K-means迭代次数
        distances_method (str): 距离计算方法
        loss_beta (float): 损失权重系数
        device (str): 计算设备
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: list,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: list,
        shared_codebook: bool,
        kmeans_method,
        kmeans_iters,
        distances_method,
        loss_beta: float,
        device: str,
    ):
        """
        初始化RQ-VAE模型

        Args:
            input_dim (int): 输入数据维度
            hidden_channels (list): 隐藏层维度列表
            latent_dim (int): 潜在空间维度
            num_codebooks (int): 码本数量
            codebook_size (list): 每个码本的大小列表
            shared_codebook (bool): 是否在所有层之间共享码本
            kmeans_method (str): K-means初始化方法
            kmeans_iters (int): K-means迭代次数
            distances_method (str): 距离计算方法
            loss_beta (float): 损失权重系数
            device (str): 计算设备
        """
        super().__init__()
        self.encoder = RQEncoder(input_dim, hidden_channels, latent_dim).to(device)
        self.decoder = RQDecoder(latent_dim, hidden_channels[::-1], input_dim).to(device)
        self.rq = RQ(
            num_codebooks,
            codebook_size,
            latent_dim,
            shared_codebook,
            kmeans_method,
            kmeans_iters,
            distances_method,
            loss_beta,
            device,
        ).to(device)

    def encode(self, x):
        """
        编码输入数据到潜在空间

        Args:
            x (torch.Tensor): 输入数据

        Returns:
            torch.Tensor: 编码后的潜在表示
        """
        return self.encoder(x)

    def decode(self, z_vq):
        """
        从量化的潜在表示解码回原始空间

        Args:
            z_vq (torch.Tensor or list): 量化后的潜在表示

        Returns:
            torch.Tensor: 解码后的数据
        """
        if isinstance(z_vq, list):
            z_vq = z_vq[-1]
        return self.decoder(z_vq)

    def compute_loss(self, x_hat, x_gt, rqvae_loss):
        """
        计算总损失，包括重构损失和量化损失

        Args:
            x_hat (torch.Tensor): 重构的数据
            x_gt (torch.Tensor): 真实数据
            rqvae_loss (torch.Tensor): RQ-VAE量化损失

        Returns:
            tuple: (recon_loss, rqvae_loss, total_loss)
                - recon_loss (torch.Tensor): 重构损失
                - rqvae_loss (torch.Tensor): 量化损失
                - total_loss (torch.Tensor): 总损失
        """
        recon_loss = F.mse_loss(x_hat, x_gt, reduction="mean")
        total_loss = recon_loss + rqvae_loss
        return recon_loss, rqvae_loss, total_loss

    def _get_codebook(self, x_gt):
        """
        获取输入数据对应的语义ID码本

        Args:
            x_gt (torch.Tensor): 输入数据

        Returns:
            torch.Tensor: 语义ID，形状为[batch_size, num_codebooks]
        """
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        return semantic_id_list

    def forward(self, x_gt):
        """
        前向传播，完整的编码-量化-解码过程

        Args:
            x_gt (torch.Tensor): 输入数据

        Returns:
            tuple: (x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss)
                - x_hat (torch.Tensor): 重构的数据
                - semantic_id_list (torch.Tensor): 语义ID
                - recon_loss (torch.Tensor): 重构损失
                - rqvae_loss (torch.Tensor): 量化损失
                - total_loss (torch.Tensor): 总损失
        """
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        x_hat = self.decode(vq_emb_list)
        recon_loss, rqvae_loss, total_loss = self.compute_loss(x_hat, x_gt, rqvae_loss)
        return x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss


if __name__ == "__main__":
    # 检测环境变量是否存在，如果不存在则尝试加载
    required_env_vars = ['TRAIN_LOG_PATH', 'TRAIN_TF_EVENTS_PATH', 'TRAIN_DATA_PATH', 'TRAIN_CKPT_PATH']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

    if missing_vars:
        try:
            from load_env import load_env
            load_env()
            print("已从 .env 文件加载环境变量")
        except ImportError:
            print('未找到 .env 文件')
    else:
        print("环境变量已存在，跳过加载")

    dataset = MmEmbDataset(data_dir=os.environ.get('TRAIN_DATA_PATH'), feature_id='85')  # 32维，本地测试下
