import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))  # key: item_reid, value: {feture_reid, value}
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}  # 这里主要用了 原始item id-> reid 的映射，因为物品的mmemb的id是原始item id
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for __, record_tuple in enumerate(user_sequence):
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))  # user profile, 每个user_seq只有一个，且一定是最后一个，所以user_profile会被insert(0)加到ext_user_sequence的最前面
                u_index = __
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))  # item interaction
        # print("len(ext_user_sequence):", len(ext_user_sequence), "u_index", u_index)

        # max_len取的是101
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)  # (102, 1)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)  # (102, 1)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)  # (102, 1)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # (102, 1)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # (102, 1)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # (102, 1)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        # nxt是行为序列中的最后一个，按照目前的也是item interaction, user profile是行为序列的第一个
        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        # ts是item interaction的item id的集合
        for record_tuple in ext_user_sequence:
            # 排除user profile
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            # i: int, item_id
            # feat: dict, 形如 {'112':14,...}
            # type_: int，按照上面的，只要是item interaction, 都是1
            # act_type: int, 0较多,1较少,少量None
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)  # dict, 形如 {'112':22, ...}
            next_feat = self.fill_missing_feat(next_feat, next_i)  # dict
            seq[idx] = i  # item_id
            token_type[idx] = type_  # item_interaction, 1
            next_token_type[idx] = next_type  # item_interaction, 1
            # next_action_type: 0, 1, 或 None
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat  # 更新feat序列
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i  # 正样本item id置为next_i
                pos_feat[idx] = next_feat  # 正样本feat置为next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)  # 负样本随机采样，不从ts里面取
                neg[idx] = neg_id  # 负样本item id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)  # 负样本feat
            nxt = record_tuple  # 更新nxt为当前record_tuple
            idx -= 1
            # idx初始值为101, -1后变成100 刚开始最多是拿 1 user_profile + 99 item_interaction 预测第 100 个item_interaction
            # 就算最长，最后一个是拿 1 user_profile 预测 第 1 个item_interaction 
            # 所以一般走不到idx==-1，这是是为了保险起见
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        # 缺失的fields可能是 sparse, array, continual, emb
        missing_fields = set(all_feat_ids) - set(feat.keys())
        # 如果不是mmemb，用默认值填充
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        # 如果是mmemb，这里应该走的不缺失的逻辑，是找到item_id对应的emb填上去；原来没有只是因为seq.jsonl里面没有这个数据
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                # 检查base_path下是否有 json 文件
                json_files = list(base_path.glob('*.json'))
                if not json_files:
                    # 如果没有json文件，则查找.part文件
                    json_files = list(base_path.glob('part*'))
                for json_file in json_files:
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict


if __name__ == '__main__':
    import os
    from main import get_args
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

    def feat_stat():
        # global dataset
        data_path = os.environ.get('TRAIN_DATA_PATH')
        args = get_args()
        dataset = MyDataset(data_path, args)
        feature_default_value, feature_types, feat_statistics = dataset._init_feat_info()

        print("=" * 80)
        print("特征信息汇总")
        print("=" * 80)

        # 1. 按类型分组显示特征统计
        print("\n📊 特征类型分布:")
        print("-" * 50)
        for feat_type, feat_ids in feature_types.items():
            if feat_ids:  # 只显示非空的特征类型
                print(f"{feat_type:15s}: {len(feat_ids):3d} 个特征 -> {feat_ids}")
            else:
                print(f"{feat_type:15s}: {len(feat_ids):3d} 个特征")

        # 2. 显示特征默认值（按类型分组）
        print(f"\n🔧 特征默认值:")
        print("-" * 50)
        for feat_type, feat_ids in feature_types.items():
            if feat_ids:
                print(f"\n{feat_type}:")
                for feat_id in feat_ids:
                    default_val = feature_default_value[feat_id]
                    if isinstance(default_val, np.ndarray):
                        print(f"  {feat_id}: numpy数组(shape={default_val.shape}, dtype={default_val.dtype})")
                    elif isinstance(default_val, list):
                        print(f"  {feat_id}: {default_val}")
                    else:
                        print(f"  {feat_id}: {default_val}")

        # 3. 显示特征统计信息（按类型分组，并排序）
        print(f"\n📈 特征统计信息:")
        print("-" * 50)
        for feat_type, feat_ids in feature_types.items():
            if feat_ids and feat_ids[0] in feat_statistics:  # 有统计信息的特征类型
                print(f"\n{feat_type}:")
                # 按统计值排序
                sorted_feats = sorted(feat_ids, key=lambda x: feat_statistics.get(x, 0), reverse=True)
                for feat_id in sorted_feats:
                    if feat_id in feat_statistics:
                        count = feat_statistics[feat_id]
                        print(f"  {feat_id}: {count:8,} 个不同值")

        # 4. 总体统计
        print(f"\n📋 总体统计:")
        print("-" * 50)
        total_features = sum(len(feat_ids) for feat_ids in feature_types.values())
        features_with_stats = len(feat_statistics)
        total_unique_values = sum(feat_statistics.values())

        print(f"总特征数量: {total_features}")
        print(f"有统计信息的特征数量: {features_with_stats}")
        print(f"所有特征的唯一值总数: {total_unique_values:,}")

        # 5. 特征规模分析
        if feat_statistics:
            print(f"\n📊 特征规模分析:")
            print("-" * 50)
            stats_values = list(feat_statistics.values())
            print(f"最大特征规模: {max(stats_values):,}")
            print(f"最小特征规模: {min(stats_values):,}")
            print(f"平均特征规模: {np.mean(stats_values):,.1f}")
            print(f"中位数特征规模: {np.median(stats_values):,.1f}")

        print("=" * 80)
    feat_stat()

