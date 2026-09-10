import hashlib


async def deterministic_reward(args, sample):
    content = f"{sample.prompt}\n{sample.response}".encode()
    return (int.from_bytes(hashlib.sha256(content).digest()[:2], "big") % 1000) / 1000.0
