# CIFAR-10 图像分类实验

正式报告见根目录 report.docx。实验比较 Custom CNN、窄通道 CIFAR ResNet-18 和带 SE 注意力的残差网络，并考察数据增强、标签平滑、权重衰减、训练轮数及 ImageNet 迁移学习。官方训练集按类别分层划分为 45,000 张训练图像和 5,000 张验证图像，划分种子为 42。测试集含 10,000 张图像。最佳权重根据验证准确率确定。

|配置|测试准确率|
|---|---:|
|正式主模型：Basic 72 轮，种子 42|89.31%|
|Basic 72 轮，三个种子|89.49% ± 0.17%|
|Cutout 72 轮，三个种子|90.29% ± 0.25%|
|ImageNet 预训练 ResNet-18，最佳第 11 轮|96.28%|

均值及样本标准差对应种子 42、43、44。最初 12 组实验包括 24 轮对照和 72 轮 Basic 复验，新增 3 组 Cutout 和 1 组迁移学习参照，共 16 组。测试集此前已经评估，新增结果属于探索性分析。预训练参照采用 ImageNet 初始化和 224×224 输入，模型容量、输入分辨率和训练预算均与从头训练配置不同。

## 数据与源码

- checkpoints 的每组目录均含 config.json、best.pt、history.json、predictions.npz 和 test.json，分别保存配置、最佳权重、逐轮日志、逐样本预测和分类指标，共 80 份核心文件。
- config 保存训练子集的标准化参数。results 包含划分索引、选模结果、复杂度、推理延迟、配对统计、Grad-CAM 热图数组与元数据。results/figures 保存实验图。
- src 包含模型、张量预处理、训练、评估与绘图程序。tests 根据归档文件检查权重加载、训练历史、预测和指标。

## 环境与结果复现

实际训练使用 Python 3.10.20、PyTorch 2.12.0.dev20260408+cu128、torchvision 0.27.0.dev20260407+cu128 和 RTX 5060 Laptop GPU。安装相互兼容的 PyTorch 与 torchvision 后，在项目根目录运行：

    python -m pip install -r requirements.txt
    python -m pytest -q
    python src/evaluate.py

测试可离线核对归档结果。evaluate 默认读取 results/selection.json 指定的最佳权重，对官方测试集计算准确率；首次运行会校验或下载 CIFAR-10 到 data。已有缓存时可通过 --data-root 指定含 cifar-10-batches-py 的目录。位置参数可指定其他从头训练的实验目录。迁移学习的模型构建与图像预处理见 src/pretrained_reference.py。

## 从头复现实验

在独立副本中运行训练，移开已有 checkpoints 后创建空的同名目录，并保留 config 和 results。首次训练自动准备 CIFAR-10；迁移学习首次运行还需下载 ImageNet 权重。按实验依赖顺序执行：

    python src/train.py --epochs 24
    python src/extended.py
    python src/weight_decay.py
    python src/extended_augmentation.py
    python src/pretrained_reference.py
    python src/gradcam_dual.py

前三条训练命令对应初始对照、Basic 长预算复验和无权重衰减对照。后续命令分别生成 Cutout 配对结果、预训练参照和样本 1884 的双目标热图。断点恢复文件由训练程序在运行时生成。平台、设备及软件版本差异可能引起数值与耗时差异。
