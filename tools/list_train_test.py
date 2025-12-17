import argparse
import json
import os
import os.path as osp
from typing import Any, Dict, List, Tuple
import re
import pandas as pd

from mmengine.config import Config, DictAction
from mmseg.utils import register_all_modules, get_classes, get_palette
import numpy as np
from PIL import Image

# 病变同义词归并（小写键） -> 标准名称（需与 metainfo classes 完全一致）
SYNONYM_MAP: Dict[str, str] = {
    # soybean rust 同义词
    'soybean leaf rust': 'soybean rust',
    'soybean_rust': 'soybean rust',
    'soybean-leaf-rust': 'soybean rust',
    'soybean leaf-rust': 'soybean rust',
    'soybean rusts': 'soybean rust',
    'soy bean rust': 'soybean rust',
    'soy-bean rust': 'soybean rust',
}


def load_metadata_mapping(data_root: str) -> Dict[str, str]:
    """加载Metadata.csv文件，创建从文件名到标准类别的映射"""
    metadata_path = osp.join(data_root, "Metadata.csv")
    if not osp.exists(metadata_path):
        print(f"警告：Metadata.csv文件不存在: {metadata_path}")
        return {}

    try:
        # 读取CSV文件
        df = pd.read_csv(metadata_path)

        # 创建文件名到类别的映射
        mapping = {}
        for _, row in df.iterrows():
            filename = row['Name']
            disease = row['Disease']
            if pd.notna(filename) and pd.notna(disease):
                # 去掉文件扩展名
                base_name = osp.splitext(filename)[0]
                mapping[base_name] = disease

        print(f"成功加载Metadata.csv，共{len(mapping)}个类别映射")
        return mapping

    except Exception as e:
        print(f"警告：加载Metadata.csv失败: {e}")
        return {}


def build_dataset_from_cfg(cfg, split: str):
    # 根据配置构建 train/test/val dataloader 对应的数据集
    assert split in ["train", "val", "test"], "split 必须是 'train'、'val' 或 'test'"
    dl_key = f"{split}_dataloader"
    assert dl_key in cfg, f"配置中缺少 {dl_key}"
    ds_cfg = cfg[dl_key]["dataset"]

    # 确保自定义 transform 被正确导入和注册
    try:
        from mmseg.datasets.transforms import (
            CopyPaste
        )
        print(f"已成功导入自定义 transform 模块")
    except ImportError as e:
        print(f"警告：无法导入自定义 transform 模块: {e}")

    # 保留完整的 pipeline，包括自定义 transform
    ds_cfg = dict(ds_cfg)

    # 保证 data_root 为绝对路径，便于输出
    data_root = ds_cfg.get("data_root", cfg.get("data_root", None))
    if data_root is None:
        raise RuntimeError("未在配置中找到 data_root")
    if not osp.isabs(data_root):
        # 相对路径基于当前工作目录（项目根目录执行脚本时更符合预期）
        data_root = osp.abspath(osp.join(os.getcwd(), data_root))
    ds_cfg["data_root"] = data_root
    from mmseg.registry import DATASETS
    # register 所有模块，防止自定义数据集不可见
    register_all_modules(init_default_scope=True)
    dataset = DATASETS.build(ds_cfg)
    # 如果数据集尚未 full_init，这里确保初始化
    if hasattr(dataset, "full_init"):
        dataset.full_init()
    return dataset


def collect_samples(dataset, split: str, use_metadata: bool = True) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    data_list = getattr(dataset, "data_list", [])
    img_prefix = dataset.data_prefix.get("img_path") if hasattr(dataset, "data_prefix") else None
    seg_prefix = dataset.data_prefix.get("seg_map_path") if hasattr(dataset, "data_prefix") else None

    # 加载Metadata映射
    metadata_mapping = {}
    if use_metadata:
        data_root = getattr(dataset, 'data_root', None)
        if data_root:
            metadata_mapping = load_metadata_mapping(data_root)

    # 构建 metainfo 类别标准名称映射（小写->原始），用于归一化回标准名
    metainfo_class_map: Dict[str, str] = {}
    if hasattr(dataset, 'metainfo') and dataset.metainfo:
        cls_list = dataset.metainfo.get('classes') or []
        if isinstance(cls_list, (list, tuple)):
            for cls in cls_list:
                if isinstance(cls, str) and cls:
                    metainfo_class_map[cls.lower()] = cls

    def normalize_class_name(name: str) -> str:
        # 去掉末尾 " (number)" 或 "_number"
        name = re.sub(r"\s*\(\d+\)$", "", name)
        name = re.sub(r"_\d+$", "", name)
        # 折叠多余空格
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def infer_class_from_path(path: str) -> str:
        # 1) 仅当路径严格为 images/<split>/<class>/<file> 才使用上一级目录为类别
        parent = osp.basename(osp.dirname(path))  # 可能是 <class> 或 <split>
        parent2 = osp.basename(osp.dirname(osp.dirname(path)))  # 可能是 <split> 或 images
        parent3 = osp.basename(osp.dirname(osp.dirname(osp.dirname(path))))  # 可能是 images
        if parent2 == split and parent3 in {"images", "imgs", "image"}:
            return normalize_class_name(parent)
        # 2) 否则从文件名推断，如 apple_scab_1.jpg -> apple_scab
        stem = osp.splitext(osp.basename(path))[0]
        parts = stem.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            base = '_'.join(parts[:-1])
        else:
            base = stem
        # 同时去掉文件名末尾的 " (number)" 和搜索引擎后缀
        base = normalize_class_name(base)
        # 处理搜索引擎后缀：_google, _Bing, _Baidu 等
        search_engines = ['google', 'Bing', 'Baidu', 'bing', 'baidu']
        for engine in search_engines:
            if base.endswith(f'_{engine}'):
                base = base[:-len(f'_{engine}')]
                break

        return base

    # 正常路径：使用数据集构建好的 data_list
    if data_list:
        for item in data_list:
            img_path = item.get("img_path") or item.get("img") or item.get("filename")
            seg_path = item.get("seg_map_path") or item.get("seg_map") or item.get("ann")
            # 转成绝对路径，若已是绝对路径则不变
            if img_path and not osp.isabs(img_path):
                if img_prefix and not osp.isabs(img_prefix):
                    img_full = osp.join(dataset.data_root or "", img_prefix, img_path)
                else:
                    img_full = osp.join(dataset.data_root or "", img_path)
            else:
                img_full = img_path
            if seg_path and not osp.isabs(seg_path):
                if seg_prefix and not osp.isabs(seg_prefix):
                    seg_full = osp.join(dataset.data_root or "", seg_prefix, seg_path)
                else:
                    seg_full = osp.join(dataset.data_root or "", seg_path)
            else:
                seg_full = seg_path
            # 特例覆盖：文件名显式包含 soybean_rust 即归为 "soybean rust"
            filename_stem = osp.splitext(osp.basename(img_full))[0]
            if re.search(r"(^|[_\s-])soybean[_\s-]?rust([_\s-]|$)", filename_stem, flags=re.IGNORECASE):
                final_class = 'soybean rust'
            # 优先使用Metadata映射获取类别
            elif metadata_mapping:
                # 从完整路径中提取文件名（不含扩展名）
                filename = filename_stem
                if filename in metadata_mapping:
                    final_class = metadata_mapping[filename]
                else:
                    # 如果Metadata中没有，回退到文件名推断
                    final_class = infer_class_from_path(img_full)
            else:
                # 没有Metadata映射，使用文件名推断
                final_class = infer_class_from_path(img_full)

            # 最终标准化：下划线->空格；先做同义词归并，再匹配 metainfo 标准名
            candidate = re.sub(r"_+", " ", str(final_class)).strip()
            key = candidate.lower()
            # 同义词归并
            if key in SYNONYM_MAP:
                candidate = SYNONYM_MAP[key]
                key = candidate.lower()
            # 匹配 metainfo 标准名
            if key in metainfo_class_map:
                final_class = metainfo_class_map[key]
            else:
                final_class = candidate

            samples.append({
                "img": img_full,
                "ann": seg_full,
                "class": final_class
            })
        return samples
    # 回退路径：data_list 为空（可能缺少标注或目录），直接扫描 images/{split}
    if img_prefix:
        img_dir = img_prefix if osp.isabs(img_prefix) else osp.join(dataset.data_root or "", img_prefix)
    else:
        # 常见默认目录
        img_dir = osp.join(dataset.data_root or "", "images", split)
    if seg_prefix:
        ann_dir = seg_prefix if osp.isabs(seg_prefix) else osp.join(dataset.data_root or "", seg_prefix)
    else:
        ann_dir = osp.join(dataset.data_root or "", "annotations", split)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    seg_suffix = getattr(dataset, "seg_map_suffix", ".png")
    if osp.isdir(img_dir):
        for name in sorted(os.listdir(img_dir)):
            ext = osp.splitext(name)[1].lower()
            if ext not in valid_exts:
                continue
            img_full = osp.join(img_dir, name)
            stem = osp.splitext(name)[0]
            ann_full = osp.join(ann_dir, stem + seg_suffix)
            if not osp.exists(ann_full):
                ann_full = None
            # 特例覆盖：文件名显式包含 soybean_rust 即归为 "soybean rust"
            filename_stem = osp.splitext(osp.basename(img_full))[0]
            if re.search(r"(^|[_\s-])soybean[_\s-]?rust([_\s-]|$)", filename_stem, flags=re.IGNORECASE):
                final_class = 'soybean rust'
            # 优先使用Metadata映射获取类别
            elif metadata_mapping:
                # 从完整路径中提取文件名（不含扩展名）
                filename = filename_stem
                if filename in metadata_mapping:
                    final_class = metadata_mapping[filename]
                else:
                    # 如果Metadata中没有，回退到文件名推断
                    final_class = infer_class_from_path(img_full)
            else:
                # 没有Metadata映射，使用文件名推断
                final_class = infer_class_from_path(img_full)

            # 最终标准化：下划线->空格；先做同义词归并，再匹配 metainfo 标准名
            candidate = re.sub(r"_+", " ", str(final_class)).strip()
            key = candidate.lower()
            # 同义词归并
            if key in SYNONYM_MAP:
                candidate = SYNONYM_MAP[key]
                key = candidate.lower()
            # 匹配 metainfo 标准名
            if key in metainfo_class_map:
                final_class = metainfo_class_map[key]
            else:
                final_class = candidate

            samples.append({
                "img": img_full,
                "ann": ann_full,
                "class": final_class
            })
    return samples


def resolve_classes(cfg) -> List[str]:
    # 优先从数据集 METAINFO 读取
    try:
        ds_type = cfg.get("dataset_type", None)
        if ds_type is None:
            # 从任意一个数据集条目里取 type
            for key in ["train_dataloader", "val_dataloader", "test_dataloader"]:
                if key in cfg and "dataset" in cfg[key] and "type" in cfg[key]["dataset"]:
                    ds_type = cfg[key]["dataset"]["type"]
                    break
        if ds_type is None:
            return []
        # 首先尝试从 utils.class_names 的别名解析
        try:
            classes = get_classes(ds_type)
            return list(classes)
        except Exception:
            pass
    except Exception:
        pass
    return []


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert('RGB')
    return img


def load_label(path: str) -> np.ndarray:
    # 读取为单通道类别索引
    label = Image.open(path)
    if label.mode != 'L':
        label = label.convert('L')
    return np.array(label, dtype=np.int32)


def ensure_dir(path: str) -> None:
    if path and not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def get_dataset_palette(dataset) -> List[List[int]]:
    # 优先从 dataset.metainfo 中取 palette
    if hasattr(dataset, 'metainfo') and dataset.metainfo and 'palette' in dataset.metainfo:
        return [list(map(int, c)) for c in dataset.metainfo['palette']]
    # 尝试从 utils.get_palette 通过别名解析
    ds_type = type(dataset).__name__
    try:
        pal = get_palette(ds_type)
        return [list(map(int, c)) for c in pal]
    except Exception:
        pass
    # 回退：生成固定色表
    rng = np.random.default_rng(1234)
    colors = rng.integers(0, 255, size=(256, 3), dtype=np.int32)
    return colors.tolist()


def render_overlay(image: Image.Image, label: np.ndarray, palette: List[List[int]], alpha: float = 0.5) -> Image.Image:
    h, w = label.shape
    color_map = np.zeros((h, w, 3), dtype=np.float32)
    unique_labels = np.unique(label)
    for cls_id in unique_labels:
        if cls_id < 0:
            continue
        color = palette[cls_id % len(palette)]
        mask = (label == cls_id)
        color_map[mask] = np.array(color, dtype=np.float32)
    img_np = np.array(image, dtype=np.float32)
    over = img_np * (1 - alpha) + color_map * alpha
    over = np.clip(over, 0, 255).astype(np.uint8)
    return Image.fromarray(over)


def save_grid(images: List[Image.Image], grid_cols: int, tile_size: Tuple[int, int], out_path: str) -> None:
    if len(images) == 0:
        return
    cols = max(1, grid_cols)
    rows = (len(images) + cols - 1) // cols
    tile_w, tile_h = tile_size
    grid = Image.new('RGB', (cols * tile_w, rows * tile_h), color=(0, 0, 0))
    for idx, img in enumerate(images):
        r_img = img.resize((tile_w, tile_h))
        r = idx // cols
        c = idx % cols
        grid.paste(r_img, (c * tile_w, r * tile_h))
    grid.save(out_path)


def visualize_split(dataset, items: List[Dict[str, Any]], out_dir: str, num: int, alpha: float, grid_cols: int = 5,
                    use_augmentation: bool = False) -> None:
    ensure_dir(out_dir)
    palette = get_dataset_palette(dataset)
    previews: List[Image.Image] = []
    limit = len(items) if num <= 0 else min(num, len(items))

    for i in range(limit):
        img_path = items[i].get("img")
        ann_path = items[i].get("ann")
        if not img_path or not osp.exists(img_path):
            continue

        if use_augmentation and hasattr(dataset, '__getitem__'):
            try:
                # 使用数据集的transform pipeline获取增强后的数据
                dataset_idx = i if i < len(dataset) else i % len(dataset)
                aug_data = dataset[dataset_idx]

                # 处理增强后的数据
                if isinstance(aug_data, dict):
                    # 如果是字典格式，提取图像和标签
                    if 'img' in aug_data:
                        img = aug_data['img']
                        if hasattr(img, 'numpy'):
                            img = img.numpy()
                        if img.ndim == 3 and img.shape[0] in [1, 3]:
                            # 转换为PIL格式
                            img = np.transpose(img, (1, 2, 0))
                            if img.shape[2] == 1:
                                img = img.squeeze(2)
                            img = Image.fromarray(img.astype(np.uint8))
                        else:
                            img = Image.fromarray(img.astype(np.uint8))
                    else:
                        img = load_image(img_path)

                    if 'gt_seg_map' in aug_data and ann_path and osp.exists(ann_path):
                        label = aug_data['gt_seg_map']
                        if hasattr(label, 'numpy'):
                            label = label.numpy()
                        if label.ndim == 3:
                            label = label.squeeze(0)
                        over = render_overlay(img, label, palette, alpha=alpha)
                    else:
                        over = img
                else:
                    # 如果返回的是元组 (img, label)
                    if isinstance(aug_data, (list, tuple)) and len(aug_data) >= 2:
                        img, label = aug_data[0], aug_data[1]
                        if hasattr(img, 'numpy'):
                            img = img.numpy()
                        if hasattr(label, 'numpy'):
                            label = label.numpy()

                        # 转换为PIL格式
                        if img.ndim == 3 and img.shape[0] in [1, 3]:
                            img = np.transpose(img, (1, 2, 0))
                            if img.shape[2] == 1:
                                img = img.squeeze(2)
                            img = Image.fromarray(img.astype(np.uint8))
                        else:
                            img = Image.fromarray(img.astype(np.uint8))

                        if label.ndim == 3:
                            label = label.squeeze(0)
                        over = render_overlay(img, label, palette, alpha=alpha)
                    else:
                        # 回退到原始图像
                        img = load_image(img_path)
                        over = img

            except Exception as e:
                print(f"警告：数据增强失败，回退到原始图像: {e}")
                img = load_image(img_path)
                over = img
        else:
            # 使用原始图像（无数据增强）
            img = load_image(img_path)
            over: Image.Image
            if ann_path and osp.exists(ann_path):
                label = load_label(ann_path)
                over = render_overlay(img, label, palette, alpha=alpha)
            else:
                # 无标注，则直接使用原图
                over = img

        # 保存单图
        base = osp.splitext(osp.basename(img_path))[0]
        aug_suffix = "_aug" if use_augmentation else ""
        out_single = osp.join(out_dir, f"{i:05d}_{base}{aug_suffix}.jpg")
        over.save(out_single)
        previews.append(over)

    # 保存网格
    if len(previews) > 0:
        aug_suffix = "_aug" if use_augmentation else ""
        grid_path = osp.join(out_dir, f"_grid{aug_suffix}.jpg")
        save_grid(previews, grid_cols=grid_cols, tile_size=(320, 240), out_path=grid_path)


def compute_class_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for it in items:
        cls_name = it.get("class")
        if not cls_name:
            # 若未能推断类别，跳过统计
            continue
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="列出训练/测试集样本与类别，并生成可视化")
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="选择导出哪个 split")
    parser.add_argument("--out", default="train_test_list.json", help="输出 JSON 文件路径")
    parser.add_argument("--max", type=int, default=0, help="仅导出前 N 条，0 表示不限制")
    parser.add_argument("--viz-dir", default="outputs/preview", help="可视化输出目录")
    parser.add_argument("--viz-num", type=int, default=25, help="每个 split 可视化数量（0 表示不限制）")
    parser.add_argument("--alpha", type=float, default=0.5, help="掩膜叠加透明度 0-1")
    parser.add_argument("--grid-cols", type=int, default=5, help="网格列数")
    parser.add_argument("--use-augmentation", action="store_true", help="在可视化时使用数据增强（训练集推荐）")
    parser.add_argument("--use-standard-classes", action="store_true", help="使用数据集标准类别定义（推荐）")
    parser.add_argument("--no-metadata", action="store_true", help="不使用Metadata.csv进行类别映射（默认使用）")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='以 xxx=yyy 覆盖配置中的字段，如 key="[a,b]" 或 key=a,b')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    result: Dict[str, Any] = {"classes": resolve_classes(cfg)}
    result_stats: Dict[str, Any] = {}

    splits = []
    if args.split == "both":
        # 默认暴露 train 和 test 两个集合
        splits = ["train", "test"]
    else:
        splits = [args.split]

    for sp in splits:
        dataset = build_dataset_from_cfg(cfg, sp)
        # 优先在首次可用时填充 classes（从数据集 METAINFO 获取）
        if not result.get("classes") and hasattr(dataset, 'metainfo'):
            ds_meta = dataset.metainfo or {}
            ds_classes = ds_meta.get('classes')
            if ds_classes:
                try:
                    result["classes"] = list(ds_classes)
                except Exception:
                    result["classes"] = [str(c) for c in ds_classes]
        items = collect_samples(dataset, sp, use_metadata=not args.no_metadata)
        if args.max and args.max > 0:
            items = items[:args.max]
        result[sp] = items
        # 统计类别计数
        result_stats[sp] = compute_class_counts(items)
        # 可视化
        split_viz_dir = osp.join(args.viz_dir, sp)
        visualize_split(dataset, items, split_viz_dir, num=args.viz_num, alpha=args.alpha, grid_cols=args.grid_cols,
                        use_augmentation=args.use_augmentation)

    # 合并统计到最终输出
    result["stats"] = result_stats
    out_path = args.out
    out_dir = osp.dirname(out_path)
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已保存到 {out_path}")
    # 控制台打印简要统计
    for sp, cnt in result_stats.items():
        total = sum(cnt.values())
        print(f"[{sp}] 总样本: {total} 类别数: {len(cnt)}")
        # 打印前若干项
        for idx, (k, v) in enumerate(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))):
            print(f"  {k}: {v}")
            if idx >= 19:
                break


if __name__ == "__main__":
    main()



