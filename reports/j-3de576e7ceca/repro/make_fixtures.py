import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"


def make_image(path: Path) -> None:
    image = Image.new("RGB", (64, 64))
    pixels = image.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = ((x * 4 + y) % 256, (x ^ y) % 256, (y * 3 + 11) % 256)
    image.save(path, format="PNG")


def row(identifier: int, text: str, has_image: bool) -> dict:
    image = str(FIXTURES / "deterministic.png") if has_image else None
    content = []
    if has_image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "The deterministic answer is seven."}]},
        ],
        "images": [image] if has_image else [],
        "metadata": {
            "fixture_id": identifier,
            "image_path": image,
            "user_text": text,
            "assistant_text": "The deterministic answer is seven.",
            "has_image": has_image,
        },
    }


def write_dataset(name: str, patterns: list[tuple[int, bool]]) -> None:
    rows = []
    for identifier, has_image in patterns:
        text = f"Fixture {identifier}: count the deterministic markers {identifier} and report the total."
        rows.append(row(identifier, text, has_image))
    with (FIXTURES / name).open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")


make_image(FIXTURES / "deterministic.png")
write_dataset(
    "data_mbs2.jsonl",
    [
        (0, False),
        (1, True),
        (2, False),
        (3, False),
        (4, True),
        (5, True),
        (6, False),
        (7, True),
    ],
)
write_dataset(
    "data_mbs4.jsonl",
    [
        *[(row_id, row_id in (2, 3)) for row_id in range(0, 4)],
        *[(row_id, False) for row_id in range(4, 8)],
        *[(row_id, True) for row_id in range(8, 12)],
        *[(row_id, row_id % 2 == 1) for row_id in range(12, 16)],
    ],
)
