"""``@Callable`` (class form) / ``session.respond(tools=...)`` demo.

Run::

    python examples/function_call.py <image_path>

What this shows:

* A ``@Callable``-decorated ``Alarm`` class with annotated fields and
  an ``__call__`` side effect.
* ``session.respond(attachment=[img], tools=[Alarm])`` — the session
  serialises the class's field schema as a tool spec, hands it to the
  model, and — when the model picks the tool — returns a *populated
  instance* of ``Alarm`` (so ``isinstance(result, Alarm)`` discriminates
  it from other tools and from the ``generating=`` payload).
* The SDK is **passive**: it never invokes the tool. The developer
  decides whether to call ``result()`` to fire the side effect.

Multi-turn agent loops (call tool → feed result back → re-query the
model) are the **developer's** responsibility. One ``respond()``
equals one model inference.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Annotated, Literal

import cv2

from wdnai import (
    Callable,
    Field,
    Generable,
    ImageAttachment,
    LLMSession,
    SystemLanguageModel,
)


# Two @Callable classes — demonstrating that ``isinstance`` discriminates
# between multiple tools without any extra carrier type.
@Callable
class Alarm:
    """Site alarm dispatcher."""
    level: Literal["low", "high"] = Field(description="事件严重等级")
    message: str = Field(description="人类可读的告警内容")

    def __call__(self):
        print(f"!!! alarm({self.level}): {self.message}")


@Callable
class Notify:
    """Push a notification to ops."""
    text: str = Field(description="通知内容")

    def __call__(self):
        print(f"[notify] {self.text}")


@Generable
class Event:
    """A structured payload the model may emit directly."""
    text: str = Field(default="无", description="事件描述文本")
    level: str = Field(default="低", description="事件等级")


async def main(image: Path) -> None:
    if not image.is_file():
        bundled = Path(__file__).resolve().parent / "demo.jpg"
        if bundled.is_file():
            image = bundled
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"图片读不到: {image}")

    model = SystemLanguageModel.qwen3_vl_4b()
    session = LLMSession(model=model)
    try:
        prompt = (
            "这是一张公共场所内的图片，如果发现火灾、人员摔倒、"
            "打架等异常情况，请调用对应工具"
        )
        session.reset()
        result = await session.respond(
            prompt=prompt,
            attachment=[ImageAttachment(img)],
            tools=[Alarm, Notify],   # both tools available
            generating=Event,        # model may emit Event JSON instead
            debug=False,
        )
        # ``result`` is one of: Alarm instance, Notify instance, Event
        # instance, or plain string. Use ``isinstance`` to dispatch.
        if isinstance(result, Alarm):
            print(f"\n[Alarm] level={result.level} message={result.message}")
            result()
        elif isinstance(result, Notify):
            print(f"\n[Notify] text={result.text}")
            result()
        elif isinstance(result, Event):
            print(f"\n[Event] text={result.text} level={result.level}")
        elif isinstance(result, str):
            print(f"\n[text] {result}")
    finally:
        session.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "image",
        nargs="?",
        default="",
        help="input image path (default: bundled examples/demo.jpg)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image = Path(args.image) if args.image else Path()
    try:
        asyncio.run(main(image))
    except KeyboardInterrupt:
        sys.exit(130)