""" Acknowledgement: This file is modified from the repository

    https://github.com/Yuriyzabegaev/FTHM-Solver

The original file was written by Yuri Zabegaev.
"""

from dataclasses import dataclass, field
from functools import cached_property
import json
from pathlib import Path
from dataclasses import asdict
import time 
import scipy.sparse as sps

import numpy as np


@dataclass
class LinearSolveStats:
    simulation_dt: float = -1
    krylov_iters: int = -1
    petsc_converged_reason: int = -100
    error_matrix_contribution: float = -1
    num_sticking: int = -1
    num_sliding: int = -1
    num_open: int = -1
    # Assumptions
    coulomb_mismatch: float = -1
    sticking_u_mismatch: float = -1
    lambdan_max: float = -1
    lambdat_max: float = -1
    un_max: float = -1
    ut_max: float = -1
    # Matrix saving
    matrix_id: str = ""
    rhs_id: str = ""
    state_id: str = ""
    iterate_id: str = ""

@dataclass
class TimeStepStats:
    linear_solves: list[LinearSolveStats] = field(default_factory=list)
    nonlinear_convergence_status: int = 1  # 1 converged -1 diverged

    @classmethod
    def from_json(cls, json: str):
        data = cls(**json)
        tmp = []
        for x in data.linear_solves:
            payload = {
                k: v for k, v in x.items() if k in LinearSolveStats.__dataclass_fields__
            }
            tmp.append(LinearSolveStats(**payload))
        data.linear_solves = tmp
        return data


def dump_json(name, data):
    save_path = Path("./stats")
    save_path.mkdir(exist_ok=True)
    try:
        dict_data = [asdict(x) for x in data]
    except TypeError:
        dict_data = data
    json_data = json.dumps(dict_data)
    with open(save_path / name, "w") as file:
        file.write(json_data)


class SolverStatistics:
    _linear_solve_stats: LinearSolveStats
    _time_step_stats: TimeStepStats

    @cached_property
    def statistics(self) -> list[TimeStepStats]:
        return []
    
    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        super().after_nonlinear_iteration(solution_vector)
        print(
            f"Newton iter: {len(self._time_step_stats.linear_solves)}, "
            f"Krylov iters: {self._linear_solve_stats.krylov_iters}"
        )
        self._linear_solve_stats.simulation_dt = self.time_manager.dt
        self._time_step_stats.linear_solves.append(self._linear_solve_stats)
        # if self.params["setup"].get("save_matrix", False):
        #     self.save_matrix_state()
        #dump_json(self.simulation_name() + ".json", self.statistics)
        #from plot_utils import write_dofs_info
        #write_dofs_info(self)

    def before_nonlinear_iteration(self) -> None:
        super().before_nonlinear_iteration()
        self._linear_solve_stats = LinearSolveStats()

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        self._time_step_stats = TimeStepStats()
        self.statistics.append(self._time_step_stats)
        print()
        DAY = 24 * 60 * 60
        print(f"Sim time: {self.time_manager.time / DAY}, Dt: {self.time_manager.dt / DAY :.2f} (days)")

    def save_matrix_state(self):
        save_path = Path("./matrices")
        save_path.mkdir(exist_ok=True)
        mat, rhs = self.linear_system
        name = f"{self.simulation_name()}_{int(time.time() * 1000)}"
        print('Saving matrix', name)
        mat_id = f"{name}.npz"
        rhs_id = f"{name}_rhs.npy"
        state_id = f"{name}_state.npy"
        iterate_id = f"{name}_iterate.npy"
        sps.save_npz(save_path / mat_id, self.bmat.mat)
        np.save(save_path / rhs_id, rhs)
        np.save(
            save_path / state_id,
            self.equation_system.get_variable_values(time_step_index=0),
        )
        np.save(
            save_path / iterate_id,
            self.equation_system.get_variable_values(iterate_index=0),
        )
        self._linear_solve_stats.iterate_id = iterate_id
        self._linear_solve_stats.state_id = state_id
        self._linear_solve_stats.matrix_id = mat_id
        self._linear_solve_stats.rhs_id = rhs_id
