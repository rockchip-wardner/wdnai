"""Qwen3-VL-4B VLM 推理示例。

模型文件位于固定的本地路径 ``/userdata/models/Qwen3-VL-4B/``：
    llm_Qwen3-VL-4B.rknn3 / .weight
    vision_Qwen3-VL-4B.rknn3 / .weight
    Qwen3-VL-4B.embed.bin
    Qwen3-VL-4B.tokenizer.gguf     ← RKNN 自带的 GGUF 格式 tokenizer

可通过环境变量更改根目录：
    RKAI_MODELS_DIR=/path/to/models
"""
from typing import Literal

import cv2

from rkai import Attachment, Generable, Callable, Field, LLMSession, SystemLanguageModel

@Generable
class Event:
    """自定义想从视频中获取到何种信息模板"""
    act: str = Field(description="图片人物行为")
    level: Literal["Normal", "Warning", "Danger", "Crime"] = Field(
        default="Normal", description="行为安全等级"
    )

@Callable
def alarm(parm: Event):
    """自定义报警函数"""
    print(f"!!! 发现异常行为，触发报警 !!! {parm}")

async def main():
    img = cv2.imread("demo.jpg")
    if img is None:
        raise FileNotFoundError("图片不存在，请放在 demo_vlm.py 同级目录")

    model = SystemLanguageModel.qwen3_vl_4b()
    session = LLMSession(model=model)
    try:
        msg = '这是一张公共场所内的图片，请检查出是否有人在车厢内做出危险行为，或者摔倒等异常情况'
        for i in range(10):
            print(f"\n========== run {i} ==========")
            # reset() 清空 history + C 库 KV cache，避免上一轮状态污染下一轮
            session.reset()
            result = await session.respond(
                msg=msg,
                attachment=Attachment(img),
                generating=Event,
                tools=[alarm],
            )
            print(f"  act  = {result.act!r}")
            print(f"  level= {result.level!r}")
    finally:
        session.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
