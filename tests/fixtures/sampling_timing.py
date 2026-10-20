"""Fixed-budget paired timing; never repeat until a desired verdict appears."""

from statistics import median
from time import perf_counter


def measure_sampling_pair(sampled, full):
    # Both paths get the same warmup. Alternate order to avoid first-path bias.
    sampled()
    full()
    timings = [[], []]
    functions = (sampled, full)
    for round_index in range(7):
        for index in (0, 1) if round_index % 2 == 0 else (1, 0):
            start = perf_counter()
            functions[index]()
            timings[index].append(perf_counter() - start)
    return median(timings[0]), median(timings[1])
