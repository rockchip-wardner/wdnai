#!/usr/bin/env python3
"""从 numpy 源码生成 C 侧的 ziggurat 常量表。

输入：numpy/random/src/distributions/ziggurat_constants.h（numpy 2.5.3）
输出：wdnai cshim 的 qi/np_zig_tables.h

⚠️ 表是 numpy 的数据（BSD-3-Clause；ziggurat 方法源自 Julia，MIT）
   ⇒ 只重新生成、别手改 qi/np_zig_tables.h。

用法：
    python3 gen_np_zig_tables.py [ziggurat_constants.h 的路径]
"""
import pathlib
import re
import sys

SRC = (sys.argv[1] if len(sys.argv) > 1
        else "/home/rockchip/qi_work/npsrc/ziggurat_constants.h")
DST = str(pathlib.Path(__file__).resolve().parent.parent / "qi" / "np_zig_tables.h")

src = open(SRC).read()


def parse_u64(name):
    m = re.search(r"static const uint64_t %s\[\] = \{(.*?)\};" % name, src, re.S)
    return [int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]+)ULL", m.group(1))]


def parse_f64(name):
    m = re.search(r"static const double %s\[\] = \{(.*?)\};" % name, src, re.S)
    body = m.group(1)
    # 去掉注释再取数字
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    return [float(t) for t in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+", body)]


ki = parse_u64("ki_double")
wi = parse_f64("wi_double")
fi = parse_f64("fi_double")
assert len(ki) == 256, len(ki)
assert len(wi) == 256, len(wi)
assert len(fi) == 256, len(fi)

NOR_R = 3.6541528853610087963519472518
NOR_INV_R = 0.27366123732975827203338247596


def fmt_rows(vals, per_row, fmt):
    out = []
    for i in range(0, len(vals), per_row):
        out.append("    " + ", ".join(fmt(v) for v in vals[i:i + per_row]) + ",")
    return "\n".join(out)


header = """// ---------------------------------------------------------------------------
// numpy 的 ziggurat 常量表（**自动生成，勿手改**）
//
//   生成脚本：wdnai/src/wdnai/_cshim/tools/gen_np_zig_tables.py
//   数据来源：numpy 2.5.3 — numpy/random/src/distributions/ziggurat_constants.h
//
// ⛔ 为什么需要它：
//    Qwen-Image-2.1 的「固定初始噪声」原本是个随模型包发布的二进制
//    （reference_512/hidden_states.bin，128 KB）。实测它是这么来的：
//
//        np.random.default_rng(0).standard_normal(64**3)
//            .reshape(64, 64, 64)[::2, ::2, :].reshape(-1).astype(np.float16)
//
//    而 numpy 的标准正态用的是 **PCG64 + ziggurat**（不是 libstdc++ 的极坐标法），
//    且 numpy 官方只保证「同一 build 内」的流一致性 —— 跨版本可能换算法
//    （官方文档举的例子就是「Box-Muller → Ziggurat」）。
//    ⇒ 用 numpy 现算 ⇒ 换个 numpy 版本 md5 就变 ⇒ 验收基线失效。
//    ⇒ 这里把 PCG64 + ziggurat 完整搬进 C 侧：**永久与该文件逐字节一致，
//      且不依赖任何外部库**。
//
// 📜 上游 license（numpy/random/src/distributions/LICENSE.md）：
//    NumPy — Copyright (c) 2005-2017, NumPy Developers. BSD-3-Clause。
//    ziggurat 方法源自 Julia — Copyright (c) 2009-2019 Jeff Bezanson 等，MIT。
//    本文件只含**数值常量表**（算法事实），保留上述声明。
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace qi {

// ki_double[i]：第 i 层的接受阈值（整数比较）
static const uint64_t NP_ZIG_KI[256] = {
%s
};

// wi_double[i]：第 i 层的宽度（x = rabs * wi_double[idx]）
static const double NP_ZIG_WI[256] = {
%s
};

// fi_double[i]：第 i 层的 pdf 值（慢路径判定用）
static const double NP_ZIG_FI[256] = {
%s
};

static const double NP_ZIG_NOR_R = %.25g;
static const double NP_ZIG_NOR_INV_R = %.25g;

}  // namespace qi
""" % (
    fmt_rows(ki, 3, lambda v: "0x%016XULL" % v),
    fmt_rows(wi, 3, lambda v: "%.25e" % v),
    fmt_rows(fi, 3, lambda v: "%.25e" % v),
    NOR_R,
    NOR_INV_R,
)

open(DST, "w").write(header)
print("已生成 %s (%d 字节)" % (DST, len(header)))
