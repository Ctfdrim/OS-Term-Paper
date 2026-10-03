#!/usr/bin/env python3
"""
CSE-307: Operating Systems — Term Paper (Track 2)
Learned Disk Scheduler Selector

Author : Anirudha Das (ID 202414098)
Course : CSE-307 Operating Systems, Spring 2026

This script implements four disk-scheduling algorithms and a small decision
tree that picks the best one for a window of pending requests:

    1. FCFS   — First-Come, First-Served
    2. SCAN   — elevator algorithm
    3. C-SCAN — circular SCAN
    4. SSTF   — Shortest Seek Time First
    5. Selector — decision tree that predicts which of the four needs the
                  least head movement, with a confidence score

The selector is tested on workloads that change type partway through
(sequential -> random -> bursty -> sequential).

Usage:
    python main.py                  # run with defaults
    python main.py --seed 7         # change random seed
    python main.py --timelines 20   # change number of test timelines
"""

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt

# ─────────────────────────────── defaults ────────────────────────────
SEED        = 42
MAX_TRACK   = 199       # disk has tracks 0..199
WINDOW_MS   = 100       # requests arriving in 100 ms form one window
PHASES      = ["sequential", "random", "bursty", "sequential"]
PHASE_LEN   = 100       # windows per phase
TIMELINES   = 10        # number of test timelines
TREE_DEPTH  = 6
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")


# ====================================================================
#  1.  DISK  SCHEDULING  ALGORITHMS
# ====================================================================
# Each function takes the head position and the requests (in arrival
# order) and returns the list of positions the head visits. Seek time is
# taken as the distance the head moves. SCAN and C-SCAN go toward track 0
# first.

def total_seek(path):
    """Total head movement along a path."""
    return sum(abs(b - a) for a, b in zip(path, path[1:]))


def fcfs(head, requests):
    """Serve requests in the order they arrived."""
    return [head] + list(requests)


def sstf(head, requests):
    """Always serve the request closest to the head."""
    pending = list(requests)
    path = [head]
    while pending:
        nearest = min(pending, key=lambda t: abs(t - path[-1]))
        pending.remove(nearest)
        path.append(nearest)
    return path


def scan(head, requests):
    """Move toward 0, go to the end, turn around and sweep upward."""
    below = sorted((t for t in requests if t <= head), reverse=True)
    above = sorted(t for t in requests if t > head)
    path = [head] + below
    if below and above and path[-1] != 0:
        path.append(0)
    return path + above


def cscan(head, requests):
    """Serve only while moving down, then jump from 0 to 199 and continue."""
    below = sorted((t for t in requests if t <= head), reverse=True)
    above = sorted((t for t in requests if t > head), reverse=True)
    path = [head] + below
    if above:
        if path[-1] != 0:
            path.append(0)
        path.append(MAX_TRACK)          # the jump counts as movement
        path += above
    return path


SCHEDULERS = {"FCFS": fcfs, "SCAN": scan, "C-SCAN": cscan, "SSTF": sstf}
NAMES = list(SCHEDULERS)                # also the tie-break order


def all_costs(head, requests):
    return [total_seek(f(head, requests)) for f in SCHEDULERS.values()]


def check_algorithms():
    """Quick check on a textbook example: head 53, queue below."""
    queue = [98, 183, 37, 122, 14, 124, 65, 67]
    expected = {"FCFS": 640, "SCAN": 236, "C-SCAN": 386, "SSTF": 236}
    got = dict(zip(NAMES, all_costs(53, queue)))
    assert got == expected, f"scheduler check failed: {got}"
    print("  Scheduler check (head 53, queue 98 183 37 122 14 124 65 67): "
          + ", ".join(f"{n} {v}" for n, v in got.items()) + "  OK")


# ====================================================================
#  2.  WORKLOAD  GENERATION
# ====================================================================

FEATURES = ["queue_depth", "track_std", "mean_jump",
            "burstiness", "head_pos", "frac_below_head"]


def make_stream(kind, n_windows, rng):
    """Arrival times (ms) and track numbers for one workload."""
    duration = n_windows * WINDOW_MS

    if kind == "bursty":
        # bursts of requests clustered around a hot track, idle gaps between
        n_bursts = rng.poisson(rng.uniform(0.01, 0.04) * duration)
        starts = np.sort(rng.uniform(0, duration, n_bursts))
        spread = rng.uniform(3, 12)
        times, tracks = [], []
        for s in starts:
            size = rng.integers(10, 41)
            center = rng.uniform(0, MAX_TRACK)
            times.append(s + np.cumsum(rng.exponential(0.05, size)))
            tracks.append(np.clip(rng.normal(center, spread, size),
                                  0, MAX_TRACK).astype(int))
        if not times:
            return np.array([]), np.array([], dtype=int)
        times, tracks = np.concatenate(times), np.concatenate(tracks)
        order = np.argsort(times)
        times, tracks = times[order], tracks[order]
        keep = times < duration
        return times[keep], tracks[keep]

    # sequential and random: requests arrive at a steady random rate
    rate = rng.uniform(0.15, 0.6)               # requests per ms
    times = np.sort(rng.uniform(0, duration, rng.poisson(rate * duration)))
    if kind == "random":
        return times, rng.integers(0, MAX_TRACK + 1, len(times))

    # sequential: 1-3 readers, each one walking along the disk
    readers = rng.integers(1, 4)
    position = rng.uniform(0, MAX_TRACK + 1, readers)
    step = rng.uniform(0.2, 1.0, readers)       # tracks per request
    who = rng.integers(0, readers, len(times))
    tracks = np.empty(len(times), dtype=int)
    for i, r in enumerate(who):
        tracks[i] = int(position[r]) % (MAX_TRACK + 1)
        position[r] += step[r]
    return times, tracks


def make_windows(kind, n_windows, rng, phase=0):
    """Cut a stream into windows. Head starts each window at a random track."""
    times, tracks = make_stream(kind, n_windows, rng)
    windows = []
    for w in range(n_windows):
        lo, hi = np.searchsorted(times, [w * WINDOW_MS, (w + 1) * WINDOW_MS])
        if hi - lo < 3:                         # too few requests to schedule
            continue
        windows.append({"kind": kind, "phase": phase,
                        "tracks": tracks[lo:hi], "times": times[lo:hi],
                        "head": int(rng.integers(0, MAX_TRACK + 1))})
    return windows


def window_features(w):
    """Numbers the classifier gets for one window."""
    tracks, head = w["tracks"], w["head"]
    gaps = np.diff(w["times"])
    return [
        len(tracks),                                          # queue depth
        tracks.std() / MAX_TRACK,                             # spread of tracks
        np.abs(np.diff(tracks)).mean() / MAX_TRACK,           # jump between arrivals
        gaps.std() / gaps.mean() if gaps.mean() > 0 else 0,   # burstiness
        head / MAX_TRACK,                                     # head position
        (tracks < head).mean(),                               # share below head
    ]


def build_dataset(windows):
    """Features, seek cost of each scheduler, best scheduler per window."""
    X = np.array([window_features(w) for w in windows])
    costs = np.array([all_costs(w["head"], [int(t) for t in w["tracks"]])
                      for w in windows])
    return X, costs, costs.argmin(axis=1)


# ====================================================================
#  3.  LEARNED  SELECTOR
# ====================================================================

def train_selector(seed):
    """Decision tree trained on windows from all three workload types."""
    rng = np.random.default_rng(seed)
    windows = []
    for kind in ["sequential", "random", "bursty"]:
        for _ in range(100):                    # 100 streams of 40 windows each
            windows += make_windows(kind, 40, rng)
    X, costs, best = build_dataset(windows)
    tree = DecisionTreeClassifier(max_depth=TREE_DEPTH, min_samples_leaf=20,
                                  random_state=seed).fit(X, best)
    share = ", ".join(f"{n} {np.mean(best == k):.0%}" for k, n in enumerate(NAMES))
    print(f"  Training windows : {len(windows)}")
    print(f"  Best scheduler   : {share}")
    print(f"  Training accuracy: {tree.score(X, best):.4f}")
    return tree


# ====================================================================
#  4.  EXPERIMENT
# ====================================================================

def run_timelines(tree, seed, n_timelines):
    """Replay timelines whose workload type changes, window by window."""
    rows = []
    for t in range(n_timelines):
        rng = np.random.default_rng(seed + 100 + t)
        windows = []
        for phase, kind in enumerate(PHASES):
            windows += make_windows(kind, PHASE_LEN, rng, phase)
        X, costs, best = build_dataset(windows)
        proba = tree.predict_proba(X)
        pick = tree.classes_[proba.argmax(axis=1)]
        for i, w in enumerate(windows):
            row = {"timeline": t, "phase": w["phase"], "kind": w["kind"],
                   "requests": len(w["tracks"])}
            row.update({n: costs[i, k] for k, n in enumerate(NAMES)})
            row.update({"Selector": costs[i, pick[i]], "Best": costs[i].min(),
                        "pick": NAMES[pick[i]], "confidence": proba[i].max(),
                        "optimal": costs[i, pick[i]] == costs[i].min()})
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(df):
    """Print and save the main results table."""
    cols = NAMES + ["Selector", "Best"]
    totals = df.groupby("timeline")[cols].sum() / 1000
    phase_mean = df.groupby("phase")[cols].mean()

    hdr = (f"  {'Policy':9s} | {'Total seek (k)':>14s} | {'Std':>5s} | "
           f"{'vs Best':>8s} | " + " | ".join(f"Phase{i + 1}" for i in range(len(PHASES))))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for c in cols:
        above = 100 * (totals[c].sum() / totals["Best"].sum() - 1)
        line = (f"  {c:9s} | {totals[c].mean():14.1f} | {totals[c].std():5.1f} | "
                f"{above:+7.1f}% | "
                + " | ".join(f"{phase_mean.loc[p, c]:6.0f}" for p in phase_mean.index))
        print(line)
        row = {"Policy": c, "Total Seek (k tracks)": round(totals[c].mean(), 1),
               "Std": round(totals[c].std(), 1), "Above Best (%)": round(above, 1)}
        row.update({f"Phase {p + 1} ({PHASES[p]})": round(phase_mean.loc[p, c])
                    for p in phase_mean.index})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"), index=False)

    wins = int((totals["Selector"] < totals["SSTF"]).sum())
    saving = 100 * (1 - totals["Selector"] / totals["SSTF"])
    print(f"\n  Selector beats always-SSTF in {wins} of {len(totals)} timelines "
          f"(saves {saving.mean():.1f}% +- {saving.std():.1f}%)")
    return totals, phase_mean


def shift_analysis(df):
    """How does the selector do right after the workload changes?"""
    position = df.groupby(["timeline", "phase"]).cumcount()
    after = (df.phase > 0) & (position < 10)
    for name, part in (("first 10 windows after a shift", df[after]),
                       ("all other windows", df[~after])):
        extra = 100 * (part["Selector"].sum() / part["Best"].sum() - 1)
        print(f"  {name:32s}: optimal choice {part['optimal'].mean():.1%}, "
              f"seek {extra:+.1f}% vs best")
    print()
    for kind in ["sequential", "random", "bursty"]:
        part = df[df.kind == kind]
        share = "  ".join(f"{n} {(part[n] <= part['Best']).mean():.0%}" for n in NAMES)
        print(f"  Best scheduler in {kind:10s} windows: {share}")


def calibration_analysis(df):
    """Is the confidence higher when the selector is right?"""
    right, wrong = df[df.optimal], df[~df.optimal]
    print(f"  Optimal choice               : {df.optimal.mean():.1%} of windows")
    print(f"  Mean confidence when optimal : {right.confidence.mean():.3f}")
    print(f"  Mean confidence when not     : {wrong.confidence.mean():.3f}")
    print(f"  AUROC of the confidence      : {roc_auc_score(df.optimal, df.confidence):.3f}")

    bins = np.clip((df.confidence * 10).astype(int), 0, 9)
    table = df.groupby(bins).agg(windows=("optimal", "size"),
                                 mean_confidence=("confidence", "mean"),
                                 share_optimal=("optimal", "mean"))
    ece = (table.windows / len(df) * (table.share_optimal - table.mean_confidence).abs()).sum()
    print(f"  Expected calibration error   : {ece:.3f}")
    table.round(3).to_csv(os.path.join(RESULTS_DIR, "calibration_table.csv"))
    return table


def paging_analysis(df):
    """What the seek times mean for virtual memory.

    If every page fault is one disk request, the effective access time is
        EAT = (1 - p) * T_mem + p * T_fault
    T_fault = controller + seek + rotation + transfer. A 7200 rpm disk is
    assumed: 0.2 ms controller, 4.17 ms rotation, 0.03 ms for a 4 KB page,
    and an 8 ms average seek, which is 0.12 ms per track (the average random
    seek is about 200/3 tracks). Memory access takes 100 ns.
    """
    ms_per_track = 8 / (200 / 3)
    rows = []
    for c in NAMES + ["Selector"]:
        tracks_per_request = df[c].sum() / df.requests.sum()
        t_fault = 0.2 + 4.17 + 0.03 + tracks_per_request * ms_per_track  # ms
        eat = lambda p: ((1 - p) * 100 + p * t_fault * 1e6)              # ns
        rows.append({"Policy": c, "Tracks per request": round(tracks_per_request, 1),
                     "Fault time (ms)": round(t_fault, 2),
                     "EAT p=1e-4 (ns)": round(eat(1e-4)), "EAT p=1e-2 (us)": round(eat(1e-2) / 1000, 1)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(RESULTS_DIR, "paging_table.csv"), index=False)


# ====================================================================
#  5.  PLOTTING
# ====================================================================

PALETTE = {"FCFS": "#7f8c8d", "SCAN": "#3498db", "C-SCAN": "#2ecc71",
           "SSTF": "#e74c3c", "Selector": "#9b59b6", "Best": "#bdc3c7"}


def make_plots(df, totals, phase_mean, calib):
    cols = NAMES + ["Selector", "Best"]

    # ── Plot 1: total seek ──
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [totals[c].mean() for c in cols]
    bars = ax.bar(cols, vals, yerr=[totals[c].std() for c in cols],
                  color=[PALETTE[c] for c in cols], edgecolor="white", capsize=3)
    ax.set_ylabel("Total seek per timeline (thousand tracks)", fontsize=12)
    ax.set_title("Total Seek by Scheduler", fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "total_seek.png"), dpi=150)
    plt.close()

    # ── Plot 2: seek per window in each phase ──
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(phase_mean))
    w = 0.8 / len(cols)
    for j, c in enumerate(cols):
        ax.bar(x + j * w, phase_mean[c], w, label=c, color=PALETTE[c])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([f"{i + 1}. {p}" for i, p in enumerate(PHASES)])
    ax.set_yscale("log")
    ax.set_ylabel("Mean seek per window (tracks, log scale)", fontsize=12)
    ax.set_title("Seek in Each Phase of the Timeline", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "seek_by_phase.png"), dpi=150)
    plt.close()

    # ── Plot 3: extra seek over the best choice, one timeline ──
    one = df[df.timeline == 0].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in ["SCAN", "C-SCAN", "SSTF", "Selector"]:
        ax.plot((one[c] - one["Best"]).cumsum() / 1000, label=c,
                color=PALETTE[c], linewidth=2.5 if c == "Selector" else 1.4)
    shifts = one.index[one.phase.diff().fillna(0) != 0]
    for i, s in enumerate(shifts):
        ax.axvline(s, color="gray", linestyle="--", linewidth=1.2,
                   label="Workload shift" if i == 0 else None)
    fcfs_extra = (one["FCFS"] - one["Best"]).sum() / 1000
    ax.set_xlabel("Window number", fontsize=12)
    ax.set_ylabel("Extra seek over best choice (thousand tracks)", fontsize=12)
    ax.set_title(f"Extra Seek Over Time (FCFS is off the chart: +{fcfs_extra:.0f}k)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "extra_seek_timeline.png"), dpi=150)
    plt.close()

    # ── Plot 4: confidence ──
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    axes[0].plot(calib.mean_confidence, calib.share_optimal, "o-",
                 color=PALETTE["Selector"], label="selector")
    axes[0].set_xlabel("Confidence", fontsize=12)
    axes[0].set_ylabel("Share of optimal choices", fontsize=12)
    axes[0].set_title("Confidence vs Accuracy", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[1].hist([df.confidence[df.optimal], df.confidence[~df.optimal]],
                 bins=np.linspace(0.3, 1, 15), stacked=True,
                 color=["#2ecc71", "#e74c3c"], label=["optimal", "not optimal"])
    axes[1].set_xlabel("Confidence", fontsize=12)
    axes[1].set_ylabel("Windows", fontsize=12)
    axes[1].set_title("Confidence of Right and Wrong Choices", fontsize=14, fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confidence.png"), dpi=150)
    plt.close()
    print("  Plots saved.")


# ====================================================================
#  6.  MAIN
# ====================================================================

def run_experiment(seed, n_timelines):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sep = "=" * 72
    print(sep)
    print("  CSE-307 Term Paper — Track 2: Learned Disk Scheduler Selector")
    print(sep)
    print(f"  Seed      : {seed}")
    print(f"  Timelines : {n_timelines} (phases: {' -> '.join(PHASES)})")
    print()

    print("[1/5] Checking the scheduling algorithms ...")
    check_algorithms()

    print("\n[2/5] Training the decision tree ...")
    tree = train_selector(seed)
    with open(os.path.join(RESULTS_DIR, "decision_tree_rules.txt"), "w") as fp:
        fp.write(export_text(tree, feature_names=FEATURES, max_depth=4))
    importance = pd.Series(tree.feature_importances_, FEATURES).sort_values(ascending=False)
    print("  Most used features: "
          + ", ".join(f"{n} {v:.2f}" for n, v in importance.head(3).items()))

    print("\n[3/5] Running the shifting workloads ...")
    df = run_timelines(tree, seed, n_timelines)
    df.to_csv(os.path.join(RESULTS_DIR, "timeline_windows.csv"), index=False)
    print(f"  {len(df)} windows tested\n")
    totals, phase_mean = summarise(df)
    print()
    shift_analysis(df)

    print("\n[4/5] Checking the confidence score ...")
    calib = calibration_analysis(df)

    print("\n[5/5] Effect on page-fault cost (virtual memory) ...")
    paging_analysis(df)

    print()
    make_plots(df, totals, phase_mean, calib)
    print(f"\n  All outputs saved to {os.path.relpath(RESULTS_DIR)}/")
    print(sep)


def parse_args():
    ap = argparse.ArgumentParser(description="Learned Disk Scheduler Selector — CSE-307 Term Paper")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    ap.add_argument("--timelines", type=int, default=TIMELINES,
                    help="Number of test timelines (default: 10)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.seed, args.timelines)
