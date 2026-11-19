"""Generate a reproducible dataset and freeze trajectory-level development/test/CV splits."""

# Run this first when we need a NEW dataset. Existing datasets go straight to training.
# handles command-line options; data.py handles the sims + saving

import argparse
from dataclasses import replace
from data import DEFAULT_DATA, generate_dataset
from src.simulation import SolverConfig


def main():
    # Steps: read options -> make solver settings -> generate states and their split IDs.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--pressure-iterations", type=int, default=50)
    parser.add_argument("--amplitude-range", type=float, nargs=2, default=(0.03, 0.15))
    parser.add_argument("--families", choices=("random", "smooth"), nargs="+", default=["random", "smooth"])
    parser.add_argument("--viscosity-range", type=float, nargs=2)
    parser.add_argument("--lid-speed-range", type=float, nargs=2)
    args = parser.parse_args()
    config = replace(SolverConfig(), n_points=args.grid_size, dt=args.dt,
                     pressure_iterations=args.pressure_iterations)
    # runs * states * 3 fields * grid area * 4 bytes per float32, converted to GiB.
    size_gib = args.runs * (args.steps + 1) * 3 * args.grid_size ** 2 * 4 / 1024 ** 3
    print(f"Uncompressed trajectory storage: approximately {size_gib:.2f} GiB", flush=True)
    try:
        # This does the expensive part; nothing is trained by this script.
        store = generate_dataset(args.output, args.runs, args.steps, args.seed, args.folds,
                                 args.test_fraction, args.split_seed, config, args.amplitude_range,
                                 args.families, args.viscosity_range, args.lid_speed_range)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Saved {len(store.trajs)} trajectories / {store.metadata['transitions']} transitions to {store.path}")
    print(f"Development: {len(store.splits['development'])}; final test: {len(store.splits['test'])}; folds: {len(store.splits['folds'])}")


if __name__ == "__main__":
    main()
