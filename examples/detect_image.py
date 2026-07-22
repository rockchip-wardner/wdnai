"""Qwen3-VL-4B VLM 推理示例。

模型文件位于固定的本地路径 ``/userdata/models/Qwen3-VL-4B/``：

    llm_Qwen3-VL-4B.1828.rknn / .weight
    vision_Qwen3-VL-4B.1828.rknn / .weight
    Qwen3-VL-4B.embed.bin
    Qwen3-VL-4B.tokenizer.gguf         ← RKNN 自带的 GGUF 格式 tokenizer

可通过环境变量更改根目录：

    RKAI_MODELS_DIR=/path/to/models

This example shows the v2 ``Attachment`` API:

* ``attachment`` is now a **list** of ``Attachment`` so one session can
  accept heterogeneous payloads (text + image + video + audio).
* ``generating`` accepts an ``@Generable`` class that inherits from
  ``LMOutput`` so the parsed result carries an ``attachment`` slot the
  SDK can populate automatically (TTS-style binaries).
"""
from typing import List, Literal

import cv2

from rkai import (
    Attachment,
    Callable,
    Field,
    Generable,
    LMOutput,
    LLMSession,
    SystemLanguageModel,
)


@Generable
class Event(LMOutput):
    """图片中抽取的结构化事件。"""
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
        prompt = (
            "这是一张公共场所内的图片，请检查出是否有人在车厢内"
            "做出危险行为，或者摔倒等异常情况"
        )
        for i in range(10):
            print(f"\n========== run {i} ==========")
            # reset() clears history + the C-level KV cache so the next
            # call doesn't see the previous turn's state.
            session.reset()
            result = await session.respond(
                prompt=prompt,
                attachment=[Attachment(img)],
                generating=Event,
                tools=[alarm],
            )
            if isinstance(result, Event):
                print(f"  act  = {result.act!r}")
                print(f"  level= {result.level!r}")
            else:
                # Model didn't emit JSON matching the schema — show the
                # raw result so the user can see what came back.
                print(f"  raw  = {result!r}")
            print(f"  out.attachment = {getattr(result, 'attachment', None)!r}")
    finally:
        session.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
