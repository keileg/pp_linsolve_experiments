import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import porepy as pp
import numpy as np
import time
from porepy.numerics.nonlinear import line_search
from typing import Optional

from stats import SolverStatistics
from FTHM_Solver.thm_solver import THMSolver

from porepy.applications.md_grids.fracture_sets import benchmark_2d_case_3


XMAX = 1000
YMAX = 1000


class Geometry:
    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": -0.1 * XMAX,
                "xmax": 1.1 * XMAX,
                "ymin": -0.1 * YMAX,
                "ymax": 1.1 * YMAX,
            }
        )

    def set_fractures(self) -> None:
        # self._fractures = []
        self._fractures = benchmark_2d_case_3(size=XMAX)


class BoundaryConditions:
    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros((self.nd, boundary_grid.num_cells))
        # rho * g * h
        # 2683 * 10 * 3000
        val = self.units.convert_units(8e7, units="Pa")
        bc_values[1, sides.north] = -val * boundary_grid.cell_volumes[sides.north]
        #  make the gradient
        bc_values[0, sides.west] = val * boundary_grid.cell_volumes[sides.west] * 1.2
        bc_values[0, sides.east] = -val * boundary_grid.cell_volumes[sides.east] * 1.2

        return bc_values.ravel("F")


class InitialCondition:
    def initial_condition(self) -> None:
        # Set initial condition for pressure, default values for other variables.
        super().initial_condition()

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        """Method returning the initial temperature values for a given grid.

        Override this method to provide different initial conditions.

        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial temperature values on that subdomain with
            ``shape=(sd.num_cells,)``. Defaults to zero array.

        """
        return np.ones(sd.num_cells) * self.reference_variable_values.temperature

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Method returning the initial pressure values for a given grid.

        Override this method to provide different initial conditions.

        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial pressure values on that subdomain with
            ``shape=(sd.num_cells,)``. Defaults to zero array.

        """
        return np.ones(sd.num_cells) * self.reference_variable_values.pressure


class Source:
    def locate_source(self, subdomains):
        source_loc_x = XMAX * 0.9
        source_loc_y = YMAX * 0.5

        ambient = [sd for sd in subdomains if sd.dim == self.nd]
        fractures = [sd for sd in subdomains if sd.dim == self.nd - 1]
        lower = [sd for sd in subdomains if sd.dim <= self.nd - 2]
        if len(self._fractures) > 0:
            x, y, z = np.concatenate([sd.cell_centers for sd in fractures], axis=1)
            source_loc = np.argmin((x - source_loc_x) ** 2 + (y - source_loc_y) ** 2)
            src_frac = np.zeros(x.size)
            src_frac[source_loc] = 1
        else:
            src_frac = np.array([])

        zeros_ambient = np.zeros(sum(sd.num_cells for sd in ambient))
        zeros_lower = np.zeros(sum(sd.num_cells for sd in lower))
        return np.concatenate([zeros_ambient, src_frac, zeros_lower])

    def fluid_source_mass_rate(self):
        if self.params["setup"]["steady_state"]:
            return 0
        else:
            return self.units.convert_units(1e1, "kg * s^-1")
            # maybe inject and then stop injecting?

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        return super().fluid_source(subdomains) + pp.ad.DenseArray(src)

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        cv = self.fluid.components[0].specific_heat_capacity
        t_inj = (
            self.units.convert_units(273 + 40, "K")
            - self.reference_variable_values.temperature
        )
        src *= cv * t_inj
        return super().energy_source(subdomains) + pp.ad.DenseArray(src)


class SolutionStrategyLocalTHM:
    def after_simulation(self):
        super().after_simulation()
        vals = self.equation_system.get_variable_values(time_step_index=0)
        name = f"thm_endstate_{int(time.time() * 1000)}.npy"
        print("Saving", name)
        self.params["setup"]["end_state_filename"] = name
        np.save(name, vals)

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual,
        reference_residual: np.ndarray,
        nl_params,
    ):
        # In addition to the standard check, print the iteration number, increment and
        # residual.
        prm = super().check_convergence(
            nonlinear_increment, residual, reference_residual, nl_params
        )
        nl_incr = self.nonlinear_solver_statistics.nonlinear_increment_norms[-1]
        res_norm = self.nonlinear_solver_statistics.residual_norms[-1]
        s = f"Non-linear increment: {nl_incr:.2e}, Residual: {res_norm:.2e}"
        print(s)

        return prm

    def compute_residual_norm(
        self, residual: Optional[np.ndarray], reference_residual: np.ndarray
    ) -> float:
        """Compute the residual norm for a nonlinear iteration.

        Parameters:
            residual: Residual of current iteration.
            reference_residual: Reference residual value (initial residual expected),
                allowing for defining relative criteria.

        Returns:
            float: Residual norm; np.nan if the residual is None.

        """
        if residual is None:
            return np.nan
        residual_norm = np.linalg.norm(residual) / np.linalg.norm(reference_residual)
        return residual_norm

    def compute_nonlinear_increment_norm(
        self, nonlinear_increment: np.ndarray
    ) -> float:
        """Compute the norm based on the update increment for a nonlinear iteration.

        Parameters:
            nonlinear_increment: Solution to the linearization.

        Returns:
            float: Update increment norm.

        """
        # Simple but fairly robust convergence criterions. More advanced options are
        # e.g. considering norms for each variable and/or each grid separately,
        # possibly using _l2_norm_cell
        # We normalize by the size of the solution vector.
        nonlinear_increment_norm = np.linalg.norm(nonlinear_increment)
        return nonlinear_increment_norm


class ConstraintLineSearchNonlinearSolver(
    line_search.ConstraintLineSearch,  # The tailoring to contact constraints.
    line_search.SplineInterpolationLineSearch,  # Technical implementation of the actual search along given update direction
    line_search.LineSearchNewtonSolver,  # General line search.
):
    """Collect all the line search methods in one class."""


class THMModel(
    Geometry,
    Source,
    InitialCondition,
    BoundaryConditions,
    # THMSolver,
    SolverStatistics,
    SolutionStrategyLocalTHM,
    pp.models.solution_strategy.ContactIndicators,
    pp.Thermoporomechanics,
):
    pass


def make_model(setup: dict):
    cell_size_multiplier = setup["grid_refinement"]

    DAY = 24 * 60 * 60

    shear = 1.2e10
    lame = 1.2e10
    if setup["steady_state"]:
        biot = 0
        dt_init = 1e-1
        end_time = 1e1
    else:
        biot = 0.47
        dt_init = 1e-3
        if setup["grid_refinement"] >= 33:
            dt_init = 1e-4
        end_time = 5e2
    porosity = 1.3e-2  # probably on the low side

    params = {
        "setup": setup,
        "folder_name": "visualization_2d_test",
        "material_constants": {
            "solid": pp.SolidConstants(
                # IMPORTANT
                permeability=1e-13,  # [m^2]
                residual_aperture=1e-3,  # [m]
                # LESS IMPORTANT
                shear_modulus=shear,  # [Pa]
                lame_lambda=lame,  # [Pa]
                dilation_angle=0 * np.pi / 180,  # [rad]
                normal_permeability=1e-4,
                # granite
                biot_coefficient=biot,  # [-]
                density=2683.0,  # [kg * m^-3]
                porosity=porosity,  # [-]
                friction_coefficient=0.577,  # [-]
                # Thermal
                specific_heat_capacity=720.7,
                thermal_conductivity=0.1,  # Diffusion coefficient
                thermal_expansion=9.66e-6,
            ),
            "fluid": pp.FluidComponent(
                compressibility=4.559 * 1e-10,  # [Pa^-1], fluid compressibility
                density=998.2,  # [kg m^-3]
                viscosity=1.002e-3,  # [Pa s], absolute viscosity
                # Thermal
                specific_heat_capacity=4182.0,  # Вместимость
                thermal_conductivity=0.5975,  # Diffusion coefficient
                thermal_expansion=2.068e-4,  # Density(T)
            ),
            "numerical": pp.NumericalConstants(
                characteristic_displacement=2e0,  # [m]
            ),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            pressure=3.5e7,  # [Pa]
            temperature=273 + 120,
        ),
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=dt_init * DAY,
            schedule=[0, end_time * DAY],
            iter_max=30,
            constant_dt=False,
        ),
        "units": pp.Units(kg=1e10),
        "meshing_arguments": {
            "cell_size": (0.1 * XMAX / cell_size_multiplier),
        },
        # experimental
        "adaptive_indicator_scaling": True,  # Scale the indicator adaptively to increase robustness
    }
    return THMModel(params)


def run_model(setup: dict):
    model = make_model(setup)
    model.prepare_simulation()
    # print(model.simulation_name())

    print("Model geometry:")
    print(model.mdg)
    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "progressbars": False,
            "nl_convergence_tol": float("inf"),
            "nl_convergence_tol_res": 1e-8,
            "nl_divergence_tol": 1e8,
            "max_iterations": 30,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 0,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 1,  # Set to 0 to use turn off the tailored line search
        },
    )

    # write_dofs_info(model)
    # print(model.simulation_name())


if __name__ == "__main__":
    common_params = {
        "solver": "CPR",
    }
    for g in [
        3,
        # 2,
        # 5,
        # 25,
        # 33,
        # 40,
    ]:
        print("Running steady state")
        params = {
            "grid_refinement": g,
            "steady_state": True,
        } | common_params
        run_model(params)
        end_state_filename = params["end_state_filename"]

        print("Running injection")
        params = {
            "grid_refinement": g,
            "steady_state": False,
            "initial_state": end_state_filename,
            "save_matrix": False,
        } | common_params
        run_model(params)
