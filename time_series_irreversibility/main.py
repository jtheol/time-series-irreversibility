import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd

from utils import (
    logistic_map,
    henon_map,
    adjacency_matrix,
    calc_outgoing_links,
    calc_async_index,
    calc_relative_async_index,
)

plt.style.use(["science", "notebook", "grid"])

if __name__ == "__main__":

    n = (2_000, 5_000, 10_000, 25_000)

    processes = {
        "gaussian": lambda size: np.random.normal(0, 1, size=size),
        "uniform": lambda size: np.random.uniform(0, 1, size=size),
        "logistic_map": logistic_map,
        "henon_map": henon_map,
    }

    results = []

    for size in n:
        for process_name, process_func in processes.items():
            data = process_func(size)
            forward, reversed = adjacency_matrix(data)

            out_forward = calc_outgoing_links(forward)
            out_reversed = calc_outgoing_links(reversed)

            ai = calc_async_index(out_forward, out_reversed)
            rai = calc_relative_async_index(out_forward, out_reversed)

            results.append(
                {
                    "process_name": process_name,
                    "size": size,
                    "async_index": ai,
                    "relative_async_index": rai,
                }
            )

    results = pd.DataFrame(results)

    fig, ax = plt.subplots(ncols=4, figsize=(20, 10))
    for idx, process in enumerate(processes.keys()):
        subset = results[results["process_name"] == process]
        ax[idx].plot(
            subset["size"], subset["relative_async_index"], label=process, marker="o"
        )
        ax[idx].set_xlabel("n")
        ax[idx].set_ylabel("RAI")
        ax[idx].set_title(f" RAI vs Size for {process}")

    plt.tight_layout()
    plt.show()
