"""Analyze MambaSeed-224 (Low-FLOPs): Params & FLOPs

- 构建 `mamba_seed_224` 模型（低 FLOPs 224 变体）
- 统计参数量 (Params)
- 使用 ptflops 计算 MACs（近似 FLOPs）
- 打印到终端，并写入本工程日志文件
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import time


@torch.no_grad()
def benchmark_infer(model, image_size=224, iters=200, warmup=50, device="cuda"):
    dev = torch.device(device)
    model.eval().to(dev)
    x = torch.randn(1, 3, image_size, image_size, device=dev)

    # warmup
    for _ in range(warmup):
        _ = model(x)
    if dev.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if dev.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters

PROJECT_MODEL_PATH = "/root/autodl-tmp/MambaSeed_224/model"
if PROJECT_MODEL_PATH not in sys.path:
    sys.path.insert(0, PROJECT_MODEL_PATH)
MV1_ROOT = "/root/autodl-tmp/MV1"
if MV1_ROOT not in sys.path:
    sys.path.insert(0, MV1_ROOT)

from mambaseed import mamba_seed_224  # noqa: F401


def build_model(
    num_classes: int = 10,
    stem_dim: int = 16,
    dims: tuple[int, ...] | None = None,
    depths: tuple[int, ...] | None = None,
):
    kwargs = {"num_classes": num_classes, "stem_dim": stem_dim}
    if dims is not None:
        kwargs["dims"] = dims
    if depths is not None:
        kwargs["depths"] = depths
    model = mamba_seed_224(**kwargs)
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_compute_flops(model: torch.nn.Module, image_size: int = 224):
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return None

    model.eval()
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            (3, image_size, image_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
    return macs


def run_visualization(device: str, num_classes: int, input_size: tuple[int, int, int, int]):
    from mambavision.tools.leaf_visualizer import LeafVisualizer

    torch_device = torch.device(device)
    model = build_model(num_classes=num_classes).to(torch_device)
    visualizer = LeafVisualizer("mamba_seed_224", input_size, num_classes, model=model)
    # benchmark model inference (ms per image)
    ms = benchmark_infer(model, image_size=input_size[-1], iters=200, warmup=50, device=device)
    print(f"Speed (ms/image): {ms:.2f}")
    # still run visualizer for outputs (not part of timing)
    visualizer.run(pretrained=False, device=torch_device)


CHANNEL_SEQUENCE = (16, 32, 64, 128, 192, 256, 512)


def module_parameter_breakdown(model: torch.nn.Module) -> dict[str, int]:
    breakdown = {"stem": count_params(model.stem)}
    for index, stage in enumerate(model.stages, start=1):
        breakdown[f"stage_{index}"] = count_params(stage)
    breakdown["norm"] = count_params(model.norm)
    breakdown["head"] = count_params(model.head)
    return breakdown


def feature_geometry(num_stages: int, image_size: int = 224) -> list[dict]:
    # The two-layer stem downsamples only in its first convolution.
    spatial = (image_size + 1) // 2
    geometry = []
    for index in range(num_stages):
        if index > 0:
            spatial = (spatial + 1) // 2
        geometry.append(
            {
                "stage": index + 1,
                "height": spatial,
                "width": spatial,
                "tokens": spatial * spatial,
            }
        )
    return geometry


def analyze_channel_prefix_sweep(
    output_dir: Path,
    image_size: int = 224,
    num_classes: int = 10,
    stem_dim: int = 16,
    measure_latency: bool = True,
    latency_iters: int = 100,
    latency_warmup: int = 30,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for length in range(1, len(CHANNEL_SEQUENCE) + 1):
        dims = CHANNEL_SEQUENCE[:length]
        depths = (1,) * length
        print(f"\nAnalyzing dims={dims}, depths={depths}", flush=True)
        model = build_model(
            num_classes=num_classes,
            stem_dim=stem_dim,
            dims=dims,
            depths=depths,
        )
        total_params = count_params(model)
        trainable_params = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        macs = try_compute_flops(model, image_size=image_size)
        geometry = feature_geometry(length, image_size=image_size)
        breakdown = module_parameter_breakdown(model)

        latency_ms = None
        if measure_latency:
            latency_ms = benchmark_infer(
                model,
                image_size=image_size,
                iters=latency_iters,
                warmup=latency_warmup,
                device=device,
            )

        record = {
            "variant": f"MambaSeed-prefix-{length}",
            "stem_dim": stem_dim,
            "dims": list(dims),
            "depths": list(depths),
            "num_stages": length,
            "num_classes": num_classes,
            "image_size": image_size,
            "expand": model.expand,
            "use_depthwise_conv": model.use_depthwise_conv,
            "use_attn": model.use_attn,
            "use_conv_branch_in_attn": model.use_conv_branch_in_attn,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": total_params - trainable_params,
            "params_m": total_params / 1e6,
            "fp32_parameter_size_mb": total_params * 4 / (1024**2),
            "macs": macs,
            "macs_g": None if macs is None else macs / 1e9,
            "latency_device": device if measure_latency else None,
            "latency_ms_per_image": latency_ms,
            "fps": None if latency_ms in (None, 0) else 1000.0 / latency_ms,
            "final_feature_resolution": (
                f"{geometry[-1]['height']}x{geometry[-1]['width']}"
            ),
            "final_token_count": geometry[-1]["tokens"],
            "feature_geometry": geometry,
            "parameter_breakdown": breakdown,
        }
        results.append(record)
        print(
            f"Params={record['params_m']:.6f} M, "
            f"MACs={record['macs_g']:.6f} G, "
            f"final={record['final_feature_resolution']}, "
            f"tokens={record['final_token_count']}, "
            f"latency={latency_ms if latency_ms is not None else 'skipped'}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = output_dir / "mambaseed_channel_prefix_detailed.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "mambaseed_channel_prefix_summary.csv"
    csv_fields = [
        "variant", "stem_dim", "dims", "depths", "num_stages",
        "total_params", "params_m", "fp32_parameter_size_mb",
        "macs", "macs_g", "final_feature_resolution", "final_token_count",
        "latency_device", "latency_ms_per_image", "fps",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in results:
            writer.writerow(
                {
                    field: (
                        tuple(record[field])
                        if field in {"dims", "depths"}
                        else record[field]
                    )
                    for field in csv_fields
                }
            )

    baseline = next(item for item in results if item["num_stages"] == 5)
    report_lines = [
        "# MambaSeed channel-prefix model analysis",
        "",
        "## Fixed configuration",
        "",
        f"- Input: `1×3×{image_size}×{image_size}`",
        f"- Classes: `{num_classes}`",
        f"- Stem dimension: `{stem_dim}`",
        "- Depth of every stage: `1`",
        f"- Expansion factor: `{baseline['expand']}`",
        f"- Depthwise local branch: `{baseline['use_depthwise_conv']}`",
        f"- Final-stage attention: `{baseline['use_attn']}`",
        (
            "- Convolution branch in final attention: "
            f"`{baseline['use_conv_branch_in_attn']}`"
        ),
        "- MACs: ptflops approximation, counted for batch size 1",
        (
            f"- Latency: {device}, batch size 1, "
            f"{latency_warmup} warm-up + {latency_iters} timed passes"
            if measure_latency
            else "- Latency: skipped"
        ),
        "",
        "## Summary",
        "",
        (
            "| Stages | dims | Final feature | Params (M) | "
            "MACs (G) | FP32 params (MiB) | Latency (ms) | FPS |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        latency_text = (
            f"{item['latency_ms_per_image']:.3f}"
            if item["latency_ms_per_image"] is not None else "N/A"
        )
        fps_text = f"{item['fps']:.2f}" if item["fps"] is not None else "N/A"
        report_lines.append(
            f"| {item['num_stages']} | `{tuple(item['dims'])}` | "
            f"{item['final_feature_resolution']} | {item['params_m']:.6f} | "
            f"{item['macs_g']:.6f} | "
            f"{item['fp32_parameter_size_mb']:.3f} | "
            f"{latency_text} | {fps_text} |"
        )

    report_lines.extend(
        [
            "",
            "## Relative to the current five-stage configuration",
            "",
            "| Stages | dims | Parameter ratio | MAC ratio |",
            "|---:|---|---:|---:|",
        ]
    )
    for item in results:
        report_lines.append(
            f"| {item['num_stages']} | `{tuple(item['dims'])}` | "
            f"{item['total_params'] / baseline['total_params']:.3f}× | "
            f"{item['macs'] / baseline['macs']:.3f}× |"
        )

    report_lines.extend(
        [
            "",
            "## Detailed parameter breakdown",
            "",
        ]
    )
    for item in results:
        report_lines.extend(
            [
                f"### {item['variant']}: `{tuple(item['dims'])}`",
                "",
                f"- Total parameters: `{item['total_params']:,}`",
                f"- Trainable parameters: `{item['trainable_params']:,}`",
                f"- Non-trainable parameters: `{item['non_trainable_params']:,}`",
                f"- Approximate MACs: `{item['macs']:,}`",
                (
                    "- Stage geometry: "
                    + " → ".join(
                        f"{entry['height']}×{entry['width']}"
                        f" ({entry['tokens']} tokens)"
                        for entry in item["feature_geometry"]
                    )
                ),
                "",
                "| Component | Parameters | Share |",
                "|---|---:|---:|",
            ]
        )
        for component, params in item["parameter_breakdown"].items():
            report_lines.append(
                f"| {component} | {params:,} | "
                f"{100 * params / item['total_params']:.2f}% |"
            )
        report_lines.append("")

    report_lines.extend(
        [
            "## Interpretation notes",
            "",
            (
                "- The final attention block always operates in the last stage. "
                "Short prefixes therefore move global attention to a much larger "
                "feature map."
            ),
            (
                "- Consequently, fewer stages do not necessarily mean fewer MACs: "
                "the one- and two-stage variants apply attention to many more tokens."
            ),
            (
                "- Parameter count, MACs, and latency describe untrained model "
                "complexity; accuracy must be measured after separately training "
                "each architecture under the same protocol."
            ),
            "",
        ]
    )
    report_path = output_dir / "MambaSeed通道前缀模型详细统计.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved {report_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MambaSeed-224 model complexity"
    )
    parser.add_argument(
        "--channel-prefix-sweep",
        action="store_true",
        help="Analyze all prefixes of (16,32,64,128,192,256,512)",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/root/autodl-tmp/MambaSeed_224/analysis/"
            "channel_prefix_sweep"
        ),
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip batch-1 inference latency benchmarking",
    )
    parser.add_argument("--latency-iters", type=int, default=100)
    parser.add_argument("--latency-warmup", type=int, default=30)
    args = parser.parse_args()

    if args.channel_prefix_sweep:
        analyze_channel_prefix_sweep(
            output_dir=Path(args.output_dir),
            measure_latency=not args.skip_latency,
            latency_iters=args.latency_iters,
            latency_warmup=args.latency_warmup,
        )
        return

    num_classes = 10
    image_size = 224
    # 所有模型参数现在自动使用模型文件中的默认值，无需手动同步

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(num_classes=num_classes).to(device)

    print(f"Current expand: {model.expand}")

    total_params = count_params(model)
    total_params_m = total_params / 1e6

    flops = try_compute_flops(model, image_size=image_size)
    if flops is not None:
        flops_g = flops / 1e9
    else:
        flops_g = None

    print("==== MambaSeed-224 Model Complexity ====")
    print(f"Num classes   : {num_classes}")
    print(f"Image size    : {image_size}x{image_size}")
    print(f"Params        : {total_params} ({total_params_m:.4f} M)")
    if flops_g is not None:
        print(f"MACs (approx) : {flops} ({flops_g:.4f} GMac)")
    else:
        print("MACs (approx) : N/A (ptflops not installed)")

    log_dir = "/root/autodl-tmp/MambaSeed_224/MambaSeed_224/output"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "MambaSeed_224_complexity.log")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("============================================\n")
        f.write(f"Time         : {now}\n")
        f.write(f"Num classes  : {num_classes}\n")
        f.write(f"Image size   : {image_size}x{image_size}\n")
        f.write(f"Params       : {total_params} ({total_params_m:.4f} M)\n")
        if flops_g is not None:
            f.write(f"MACs (approx): {flops} ({flops_g:.4f} GMac)\n")
        else:
            f.write("MACs (approx): N/A (ptflops not installed)\n")
        f.write("\n")

    print(f"Complexity info has been appended to: {log_path}")

    run_visualization(device=device.type, num_classes=num_classes, input_size=(1, 3, image_size, image_size))


if __name__ == "__main__":
    main()
