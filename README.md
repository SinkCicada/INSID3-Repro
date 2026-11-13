# INSID3 Repro

课程复现项目。以 [visinf/INSID3 的 0c165a1 提交](https://github.com/visinf/INSID3/tree/0c165a10cf52ab91f335883d06260de86854adbe) 为基础，保留官方模型、加载器和指标实现，并提供分折评测、消融、语义对应和 CWFID 应用脚本。论文发表于 CVPR 2026。作者、原始使用说明与引用见 [上游 README](README.upstream.md)。

## 安装与推理

复现实验使用 Linux、Python 3.10、PyTorch 2.7.1、CUDA 12.6 和 NVIDIA A10。带 CRF 的评测需要与 PyTorch CUDA 版本匹配的工具链。

~~~bash
git clone https://github.com/SinkCicada/INSID3-Repro.git
cd INSID3-Repro
uv sync
source .venv/bin/activate
mkdir -p data pretrain output ../downloads ../setup-logs
python download_weights.py
~~~

默认从 [GitHub Release](https://github.com/SinkCicada/INSID3-Repro/releases/tag/weights-v1) 获取主评测使用的 DINOv3 Large。全部实验权重可用 `python download_weights.py --all` 获取，分卷会自动合并，下载后校验 SHA-256。已有同名权重放在 `pretrain/` 即可复用。模型的原始许可与来源见 [权重许可](licenses/README.md)。

仓库自带猫示例图片。使用下面的命令准备 DINOv3 源码，然后执行单图推理。已具备缓存时可跳过 clone。

~~~bash
hub_dir="$(python -c 'import torch; print(torch.hub.get_dir())')"
mkdir -p "$hub_dir"
git clone https://github.com/facebookresearch/dinov3.git "$hub_dir/facebookresearch_dinov3_main"
python reproduce.py check
python reproduce.py demo --gpu 0
~~~

在终端打印的输出目录中打开 `prediction.png` 和 `predicted_mask.png`。可用 `--reference-image`、`--reference-mask` 和 `--target-image` 指定其他图片。

## 分割评测

按 [上游数据准备说明](docs/data.md) 准备 `data/`。分割命令从仓库根目录运行。GPU 0 只是示例。发布的主结果使用固定 seed 0，输入 1024，CRF 640，SVD rank 500，聚类阈值 0.6，聚合阈值 0.2。

~~~bash
CUDA_VISIBLE_DEVICES=0 python inference_segmentation.py \
  --dataset isic --model-size large --shots 1 --image-size 1024 \
  --crf-mask-refinement --crf-size 640 --svd-comps 500 \
  --tau 0.6 --merge-thresh 0.2 --seed 0 --num-workers 0
~~~

| 数据集参数 | 折编号 | 可用 shot |
|---|---|---|
| `suim`、`lung`、`isic` | 0 | 1、5 |
| `isaid` | 0–2 | 1、5 |
| `coco`、`pascal_part`、`paco_part` | 0–3 | 1、5 |
| `lvis` | 0–9 | 1、5 |
| `permis` | 0 | 1 |

使用 `--fold N` 逐折评测，对各折 mIoU 取等权平均。批量任务、应用和额外依赖见 [复现实验命令](docs/experiments.md)。

## 实测结果

- Table 1 的九个 1-shot 基准平均 mIoU 为 **55.06%**，论文为 **55.1%**。见 [逐项分数](results/table1.json)。
- 八个 5-shot 基准和分割消融见 [分割对照结果](results/segmentation-controls.json)。ISIC 的分辨率和种子对照保存在 [results](results/)。
- Figure 7 的 **128 个分折任务全部完成**。见 [扫描结果](results/figure7.json) 和 [曲线](results/figure7.png)。
- SPair-71k、替换骨干、SAM 3、空掩码和计时见 [补充对照](results/supplement.json)。其中空掩码、部分替换骨干及阈值基线和论文差距较大。SAM 3 图像横拼采用框提示。
- CWFID 的 21 张测试图以 002 为固定参考。汇总前景 IoU 为 **78.82%**，覆盖率 MAE 为 **0.54 个百分点**。见 [应用汇总](results/cwfid.json) 和 [逐图结果](results/cwfid-per-image.csv)。

完整数据集按各上游来源获取。代码许可证见 [LICENSE](LICENSE)。
