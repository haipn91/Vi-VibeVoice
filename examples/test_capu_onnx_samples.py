#!/usr/bin/env python3
"""
Kiem thu CAPU (ONNX): van ban nhieu y — chu thuong, khong dau cau (kieu ASR tho).
Chay tu goc repo:
  pip install -r python/requirements-capu.txt
  set GIPFORMER_CAPU_ONNX=1
  python examples/test_capu_onnx_samples.py

Tuy chon: python examples/test_capu_onnx_samples.py --onnx path/to/capu-seq2labels.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = Path(__file__).resolve().parent
ONNX_INT8 = EXAMPLES / "onnx" / "capu-seq2labels.int8.onnx"
ONNX_FP32 = EXAMPLES / "onnx" / "capu-seq2labels.onnx"

# Mau: nhieu cau / nhieu vong nhung noi lien, khong dau cau, chu thuong
SAMPLES: list[dict[str, str | int]] = [
    {
        "id": "bao_cao_chi_bo",
        "text": (
            "tháng ba năm hai không hai sáu chi bộ đã tập trung lãnh đạo phòng thực hiện các nội dung chuyên môn "
            "trong bối cảnh tiếp tục thực hiện nhiều nhiệm vụ đan xen trong đó có một số nhiệm vụ mới "
            "chi bộ đã tập trung lãnh đạo thực hiện toàn diện các mặt công tác tập trung vào một số nhiệm vụ trọng tâm"
        ),
        "min_new_punct": 3,
    },
    {
        "id": "ke_hoach_khcn",
        "text": (
            "lãnh đạo tham gia tốt công tác tham mưu góp ý vào sáu chương trình kế hoạch đề án dự án "
            "khoa học công nghệ đơn vị đã phối hợp với các ban ngành rà soát tiến độ và đề xuất nguồn lực "
            "cho giai đoạn tới đây là ưu tiên số một của ban chỉ đạo"
        ),
        "min_new_punct": 2,
    },
    {
        "id": "phan_mem_du_lieu",
        "text": (
            "lãnh đạo tiếp tục tham gia thực hiện xây dựng ba phần mềm tổng hợp thông tin báo cáo triển khai "
            "tập huấn thử nghiệm và hiệu chỉnh phần mềm quản lý dữ liệu đảng viên các bước sẽ hoàn thành "
            "theo đúng tiến độ được giao trong quý hiện nay"
        ),
        "min_new_punct": 2,
    },
    {
        "id": "giao_thuong_ngoai_giao",
        "text": (
            "theo đó thủ tướng dự kiến tiếp bộ trưởng nông nghiệp mỹ tom vilsack bộ trưởng thương mại mỹ gina raimondo "
            "và bộ trưởng tài chính janet yellen cuộc gặp diễn ra nhằm thúc đẩy hợp tác kinh tế song phương "
            "giữa hai nước trong các lĩnh vực then chốt"
        ),
        "min_new_punct": 4,
    },
    {
        "id": "hai_doan_ngan",
        "text": (
            "đơn vị đã hoàn thành báo cáo tổng kết năm học hội đồng khoa học sẽ họp vào tuần tới "
            "để thông qua danh mục đề tài cấp cơ sở"
        ),
        "min_new_punct": 2,
    },
    {
        "id": "nhiem_vu_ke_hoach",
        "text": (
            "phòng hành chính phối hợp với phòng tổ chức rà soát hồ sơ cán bộ và cập nhật cơ sở dữ liệu nội bộ "
            "mục tiêu là đảm bảo minh bạch kịp thời và phục vụ công tác quy hoạch đến cuối quý"
        ),
        "min_new_punct": 2,
    },
    {
        "id": "y_kien_chi_dao",
        "text": (
            "bí thư nhấn mạnh các đồng chí cần nêu cao tinh thần trách nhiệm trong thực thi nhiệm vụ "
            "tránh tình trạng chậm trễ ảnh hưởng đến uy tín của tập thể lãnh đạo địa phương"
        ),
        "min_new_punct": 2,
    },
    {
        "id": "ket_thuc_hoi_nghi",
        "text": (
            "sau các thảo luận hội nghị thống nhất giao ban tổ chức thực hiện và báo cáo kết quả "
            "bằng văn bản trước ngày mười lăm tháng tới văn phòng tổng hợp sẽ theo dõi tiến độ"
        ),
        "min_new_punct": 2,
    },
]


def _count_punct(s: str) -> int:
    return sum(s.count(c) for c in ".,:;!?…")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test CAPU ONNX on unpunctuated lowercase Vietnamese")
    parser.add_argument("--onnx", type=Path, default=None, help="Path to capu-seq2labels.onnx")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ["GIPFORMER_CAPU_ONNX"] = "1"
    onnx_p = args.onnx or (ONNX_INT8 if ONNX_INT8.is_file() else ONNX_FP32)
    os.environ["GIPFORMER_CAPU_ONNX_PATH"] = str(onnx_p.resolve())

    if not onnx_p.is_file():
        print("ERROR: Khong tim thay ONNX:", onnx_p, file=sys.stderr)
        print("  Chay: python scripts/export_capu_onnx.py [va scripts/quantize_capu_onnx.py]", file=sys.stderr)
        return 2

    sys.path.insert(0, str(EXAMPLES))
    # Import sau khi set bien moi truong
    import vi_capu_onnx as vco  # noqa: E402

    vco.load_onnx_capu()
    if not vco.onnx_capu_ready():
        err = vco.onnx_capu_last_error() or "unknown"
        print("ERROR: Khong nap duoc CAPU ONNX:", err, file=sys.stderr)
        return 3

    print("ONNX:", onnx_p)
    print("Mau:", len(SAMPLES))
    print("-" * 72)

    ok = 0
    for row in SAMPLES:
        sid = str(row["id"])
        raw = str(row["text"]).strip()
        assert raw == raw.lower(), f"{sid}: mau phai lowercase"
        min_p = int(row["min_new_punct"])

        out = vco.apply_onnx_capu_text(raw)
        pin, pout = _count_punct(raw), _count_punct(out)
        delta = pout - pin
        upper_ok = any(ch.isupper() for ch in out)

        passed = delta >= min_p and upper_ok
        if passed:
            ok += 1
        status = "OK" if passed else "FAIL"

        print(f"[{status}] {sid}  (+dau: {delta} >= {min_p}, chu_hoa: {upper_ok})")
        if args.verbose or not passed:
            print("  IN :", raw[:220] + ("…" if len(raw) > 220 else ""))
            print("  OUT:", out[:280] + ("…" if len(out) > 280 else ""))
            print()

    print("-" * 72)
    print(f"Ket qua: {ok}/{len(SAMPLES)} mau dat tieu chi (them toi thieu dau cau + co chu hoa)")
    if ok < len(SAMPLES):
        print("Ghi chu: min_new_punct la ng uong thu; mo hinh co the khop it hon tren mot so cau.")
    return 0 if ok == len(SAMPLES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
