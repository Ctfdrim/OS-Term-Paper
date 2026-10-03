#!/usr/bin/env python3
"""
CSE-307: Operating Systems — Term Paper (Track 2)
Small Linux disk test: sequential vs random reads, cold vs warm page cache.

Usage:
    python3 linux_test.py                  # test file goes in /var/tmp
    python3 linux_test.py --dir /home/me   # use another folder (must be on a real disk, not tmpfs)

It writes a 256 MB file, reads 4 KB blocks from it and prints speed and
latency (mean +- std over 5 runs). Only uses the standard library.
"""
import argparse
import mmap
import os
import platform
import random
import statistics
import time

BLOCK   = 4096          # bytes per read
FILE_MB = 256
READS   = 20000
RUNS    = 5
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def drop_from_cache(path):
    """Remove the file from the page cache so the next read goes to the disk."""
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def read_blocks(path, offsets, direct):
    """Read one block at every offset. Returns (MB/s, latency in microseconds).

    direct=True uses O_DIRECT, which skips the page cache. The buffer has to be
    page aligned, which is why mmap is used.
    """
    fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
    buf = mmap.mmap(-1, BLOCK)
    start = time.perf_counter()
    for off in offsets:
        os.preadv(fd, [buf], off)
    seconds = time.perf_counter() - start
    os.close(fd)
    return len(offsets) * BLOCK / seconds / 1e6, seconds / len(offsets) * 1e6


def repeat(test):
    runs = [test() for _ in range(RUNS)]
    speed = [r[0] for r in runs]
    latency = [r[1] for r in runs]
    return (statistics.mean(speed), statistics.stdev(speed),
            statistics.mean(latency), statistics.stdev(latency))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/var/tmp")
    path = os.path.join(ap.parse_args().dir, "linux_test_file.bin")

    with open(path, "wb") as f:
        for _ in range(FILE_MB):
            f.write(os.urandom(1 << 20))
        f.flush()
        os.fsync(f.fileno())

    blocks = FILE_MB * (1 << 20) // BLOCK
    rng = random.Random(1)
    sequential = [i * BLOCK for i in range(READS)]
    scattered = [rng.randrange(blocks) * BLOCK for _ in range(READS)]
    cache_offsets = scattered[:5000]

    def direct(offsets):
        def test():
            drop_from_cache(path)
            return read_blocks(path, offsets, True)
        return test

    def cold():
        drop_from_cache(path)
        return read_blocks(path, cache_offsets, False)

    def warm():
        drop_from_cache(path)
        read_blocks(path, cache_offsets, False)         # first pass fills the cache
        return read_blocks(path, cache_offsets, False)  # second pass comes from RAM

    results = [("direct, sequential", repeat(direct(sequential))),
               ("direct, random", repeat(direct(scattered))),
               ("buffered, cold cache", repeat(cold)),
               ("buffered, warm cache", repeat(warm))]
    os.remove(path)

    out = [f"{platform.node()}: {platform.platform()}, Python {platform.python_version()}",
           f"{READS} reads of 4 KB from a {FILE_MB} MB file, mean +- std over {RUNS} runs", "",
           f"{'test':24s}{'MB/s':>18s}{'latency (us)':>20s}"]
    for name, (s, s_sd, l, l_sd) in results:
        out.append(f"{name:24s}{s:11.1f} +-{s_sd:5.1f}{l:13.1f} +-{l_sd:5.1f}")
    print("\n".join(out))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "linux_test.txt"), "w") as fp:
        fp.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
