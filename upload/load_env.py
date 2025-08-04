import os
from pathlib import Path


def load_env(env_file='.env'):
    """
    从.env文件加载环境变量

    Args:
        env_file (str): .env文件路径，默认为当前目录下的.env文件
    """
    env_path = Path(env_file)

    if not env_path.exists():
        print(f"警告: 环境变量文件 {env_file} 不存在")
        return

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue

            # 解析键值对
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # 移除值两端的引号（如果有的话）
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]

                # 设置环境变量
                os.environ[key] = value
                print(f"已加载环境变量: {key} = {value}")


if __name__ == '__main__':
    # 直接运行此文件时加载环境变量
    load_env()
