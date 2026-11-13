# 复现实验命令

以下命令在仓库根目录、已激活的 Linux 环境运行。先完成 [安装与数据准备](../README.md)。`../setup-logs` 和 `../downloads` 是批量脚本的工作目录。单折的官方 CLI 只需要准备好的数据与权重，输出在 `output/`。

## 批量评测和对照

~~~bash
# 验证本地数据并写入 5-shot 检查记录
python run_extended_reproduction.py preflight

python run_extended_reproduction.py five-shot --gpu 0
python run_extended_reproduction.py key-ablations --gpu 0
python run_extended_reproduction.py threshold-baseline --gpu 0
python run_figure7_controls.py --experiment threshold-control --gpu 0
python run_figure7_controls.py --experiment figure7 --gpu 0
python run_table5_ablation.py --gpu 0

# SPair-71k 数据按 docs/data.md 准备
python run_spair_backbone_controls.py --gpu 0

# 同一 ISIC 配对上比较评分网格和 CRF 网格
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python isic_audit/run_grid_audit.py \
  --seed 0 --paired-grids --output output/isic-grid-seed0
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python isic_audit/seed_sweep.py \
  --seed 1 --output output/isic-seed1
~~~

Figure 7 的 rank=0 点使用关键消融的无去偏结果，因此先运行关键消融。分折任务及结果留在 `../setup-logs/` 和 `output/`。开始另一轮同名批量任务前，归档先前状态文件。

对于旧版 ISIC 对照，将官方 [26759bd 提交](https://github.com/visinf/INSID3/tree/26759bd9a98490b43882e51a32e5e8a3a6605086) 放在独立目录。运行时将该目录设为当前目录和 `PYTHONPATH`，在该目录挂载已准备的 `data/` 和 `pretrain/`，使用本仓库的 `isic_audit/run_pre_speedup.py --seed 0 --output <输出目录>`。

## 农田植被分割

将 [CWFID 数据](https://github.com/cwfid/dataset) 的 master ZIP 放在 `../downloads/cwfid-master.zip`。准备脚本会核对归档，转换掩码，并固定参考图 002。

~~~bash
python prepare_cwfid_data.py
python run_cwfid_application.py --gpu 0
~~~

输出包含 `result.json`、`per_image.csv`、预测掩码与对照图。

## 需要额外依赖的补充实验

外部代码放在 `../downloads/<仓库名>-<提交>`，权重放在 `pretrain/`。各脚本顶部的路径与 SHA-256 常量对应本次使用的文件。外部依赖按其仓库说明安装，必要时为每个基线单独建环境。

| 实验 | 入口与依赖 |
|---|---|
| Table 4、Table 8 | `python run_dinov2_controls.py table4 --gpu 0`。其余模式为 `table8-dinov2`、`table8-franca`、`table8-pe`。分别需要 DINOv2、Franca 或 Perception Encoder 的源码与对应权重。 |
| Table 9 | `python run_table9_sam3.py --mode video-mask --gpu 0` 或 `--mode image-box`。需要 SAM 3 及官方 `sam3.pt`。 |
| Table 10 | `python run_supplement_evaluation.py table10 --gpu 0`。需要 COCO 2014 的官方实例标注归档，命名为 `../downloads/coco-annotations-trainval2014.zip`。 |
| Table 11 | `python run_table11_gfsam.py --method gfsam --gpu 0`。另外支持 `matcher`、`persam`、`seggpt`。需要对应基线源码、权重及其依赖。 |
| Table 12–13 | `python run_supplement_evaluation.py table12-13 --gpu 0`。使用已准备的 COCO 折。 |

`prepare_*.py` 中的本次数据整理程序会检查原始归档和相应的校验记录。新机器可按 [上游数据说明](data.md) 直接获得可供加载器读取的布局。已备数据可以直接用 README 的官方单折入口验证。
