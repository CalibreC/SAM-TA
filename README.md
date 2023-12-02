# SAM-TA

## Install
1. 配置conda环境
```bash
conda create -n SAM-TA python=3.10 -y     
conda activate SAM-TA
```

2. 安装Poetry
```bash
conda install -c conda-forge poetry  
```

```bash
poetry install
```

PS:WSL2 Ubuntu22.04 需要添加如下环境变量

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

有关于SSH登录以及Bash环境变量设置请参考
[ssh连接远程主机执行脚本的环境变量问题](https://blog.csdn.net/vivianXuejun/article/details/80557287),
[pycharm ssh远程解释器连接docker容器环境变量缺失](https://blog.csdn.net/Farm_Coder/article/details/122212169)与
[shell登录、非登录,交互、非交互 概念详解](https://blog.csdn.net/u013259665/article/details/128856874)
