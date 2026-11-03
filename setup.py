"""wdnai 的构建脚本（setuptools）。

为什么需要它：文生图（Qwen-Image-2.1）的运行时是 **C 实现**（``wdnai/_cshim``）——
纯 Python 重写虽然也能逐字节对齐，但要把编译器行为（FMA 收缩）、fp16 次正规数
的怪癖、libstdc++ 的 mt19937 全部反推一遍（实测踩了三个 1-ULP 的坑）。

⇒ 这里在 ``build_py`` 阶段顺带把 cshim 编出来（**不需要 Python.h**，
   直接 ``make`` 一个普通 .so），装完即可用。

⚠️ 编译失败**不会**让安装失败（其它后端不依赖 cshim），
   只会在装完后提示「文生图不可用」。
"""
from __future__ import annotations

import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
CSHIM_DIR = os.path.join(HERE, "src", "wdnai", "_cshim")


class BuildCShim(build_py):
    """``build_py`` 之前先把 ``wdnai/_cshim/libqi_qwen.so`` 编出来。"""

    def run(self) -> None:  # noqa: D102
        if os.path.isdir(CSHIM_DIR):
            try:
                subprocess.check_call(["make", "-C", CSHIM_DIR])
            except (OSError, subprocess.CalledProcessError) as exc:
                sys.stderr.write(
                    "\n[wdnai] ⚠️ cshim 编译失败（{}）\n"
                    "     ⇒ 文生图（Qwen-Image-2.1）不可用，"
                    "其它后端不受影响。\n"
                    "     修好依赖后重跑： cd src/wdnai/_cshim && make\n\n".format(exc)
                )
        super().run()


setup(cmdclass={"build_py": BuildCShim})
