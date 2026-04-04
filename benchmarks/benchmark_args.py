"""
Shared argparse helpers for benchmark scripts.
Omitted flags use defaults inside utils.algorithm_adapter run_* functions.
"""
import argparse
from typing import Any, Dict


def add_algorithm_args(parser: argparse.ArgumentParser) -> None:
    """Register optional algorithm tuning flags (default=SUPPRESS when not passed)."""
    parser.add_argument(
        "--bf-update-frequency",
        type=int,
        default=argparse.SUPPRESS,
        dest="update_frequency",
        metavar="N",
        help="Bruteforce: visualization update frequency (benchmarks use visualize=False; rarely needed)",
    )
    parser.add_argument(
        "--sa-initial-temp",
        type=float,
        default=argparse.SUPPRESS,
        dest="sa_initial_temp",
        help="Simulated annealing: initial temperature",
    )
    parser.add_argument(
        "--sa-cooling-rate",
        type=float,
        default=argparse.SUPPRESS,
        dest="sa_cooling_rate",
        help="Simulated annealing: cooling rate per iteration",
    )
    parser.add_argument(
        "--sa-max-iter",
        type=int,
        default=argparse.SUPPRESS,
        dest="sa_max_iter",
        help="Simulated annealing: maximum iterations",
    )
    parser.add_argument(
        "--qaoa-layers",
        type=int,
        default=argparse.SUPPRESS,
        dest="qaoa_layers",
        help="QAOA: number of layers",
    )
    parser.add_argument(
        "--qaoa-learning-rate",
        type=float,
        default=argparse.SUPPRESS,
        dest="qaoa_learning_rate",
        help="QAOA: optimizer learning rate",
    )
    parser.add_argument(
        "--qaoa-steps",
        type=int,
        default=argparse.SUPPRESS,
        dest="qaoa_optimization_steps",
        help="QAOA: optimization steps",
    )
    parser.add_argument(
        "--num-approx",
        type=int,
        default=argparse.SUPPRESS,
        dest="qaoa_num_approx",
        help="QAOA: ApproxTimeEvolution steps (0 = exact evolve in circuit)",
    )


def algorithm_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build kwargs for run_bruteforce / run_mst / run_sim_annealing / run_qaoa."""
    keys = (
        "update_frequency",
        "sa_initial_temp",
        "sa_cooling_rate",
        "sa_max_iter",
        "qaoa_layers",
        "qaoa_learning_rate",
        "qaoa_optimization_steps",
        "qaoa_num_approx",
    )
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}
