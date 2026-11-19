#!/usr/bin/env python3
"""Patch only the known CARLA 0.9.15 binary's moved-from sensor stream crash.

ASensor::EndPlay dereferences DataStream::_shared_state after
FActorSensorData::RecordActorData moves the stream during actor dormancy.
Skip CloseStream only for a null shared state; always continue with the
original SensorManager deregistration. Never modify the input executable.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import tempfile
from pathlib import Path

SOURCE_SHA256 = "1da989e4ff136e0658fdea421bc1da2b0e5cb018c6e767efc4f6129c21ab1a81"
# This build's .text PT_LOAD has p_vaddr - p_offset = 0x201000.
TEXT_DELTA = 0x201000
HOOK = 0x27D094E          # mov 0x240(%r12), %rsi in ASensor::EndPlay
RESUME = 0x27D0956        # add $8, %rsi (original non-null path)
DEREGISTER = 0x27D0980    # SensorManager.DeRegisterSensor (null path)
CAVE = 0x6386668          # 32 bytes of INT3 padding after EC_GFp_nistz256_method
ORIGINAL_HOOK = bytes.fromhex("498bb42440020000")
ORIGINAL_CAVE = b"\xcc" * 32


def relative_jump(opcode: bytes, at: int, destination: int) -> bytes:
    return opcode + struct.pack("<i", destination - (at + len(opcode) + 4))


def build_patch(original: bytes) -> bytes:
    digest = hashlib.sha256(original).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Unsupported CARLA executable SHA256: {digest}; expected {SOURCE_SHA256}")
    hook_offset, cave_offset = HOOK - TEXT_DELTA, CAVE - TEXT_DELTA
    if original[hook_offset:hook_offset + len(ORIGINAL_HOOK)] != ORIGINAL_HOOK:
        raise ValueError("ASensor::EndPlay instruction differs; refusing patch")
    if original[cave_offset:cave_offset + len(ORIGINAL_CAVE)] != ORIGINAL_CAVE:
        raise ValueError("INT3 code cave differs; refusing patch")

    # Restore the displaced instruction, test its shared-state pointer, then
    # branch to the original deregistration or the original GetToken path.
    guarded_stream = ORIGINAL_HOOK + bytes.fromhex("4885f6")
    skip_close = relative_jump(b"\x0f\x84", CAVE + len(guarded_stream), DEREGISTER)
    resume = relative_jump(b"\xe9", CAVE + len(guarded_stream) + len(skip_close), RESUME)
    trampoline = guarded_stream + skip_close + resume
    assert len(trampoline) <= len(ORIGINAL_CAVE)
    hook = relative_jump(b"\xe9", HOOK, CAVE).ljust(len(ORIGINAL_HOOK), b"\x90")
    patched = bytearray(original)
    patched[hook_offset:hook_offset + len(ORIGINAL_HOOK)] = hook
    patched[cave_offset:cave_offset + len(trampoline)] = trampoline
    return bytes(patched)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="unaltered CARLA 0.9.15 shipping ELF")
    parser.add_argument("output", type=Path, help="new ELF in a separate CARLA runtime tree")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.output.is_symlink():
        parser.error("output must be a separate regular file, never the original or a symlink")
    patched = build_patch(args.source.read_bytes())
    output_digest = hashlib.sha256(patched).hexdigest()
    if args.output.exists():
        if hashlib.sha256(args.output.read_bytes()).hexdigest() != output_digest:
            parser.error("output already exists with different content; refusing overwrite")
        print(f"Already patched: {args.output} SHA256={output_digest}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".carla-patch-", dir=args.output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(patched)
        os.chmod(temporary, args.source.stat().st_mode & 0o777)
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Patched CARLA executable: {args.output} SHA256={output_digest}")


if __name__ == "__main__":
    main()
