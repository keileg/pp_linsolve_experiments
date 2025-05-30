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
import FTHM_Solver

from porepy.models.constitutive_laws import (
    ConstantPermeability,
    CubicLawPermeability,
    DimensionDependentPermeability,
    SpecificStorage,
)


from porepy.applications.md_grids.fracture_sets import benchmark_2d_case_3


XMAX = 2000
YMAX = 2000


class Geometry:
    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": 0,
                "xmax": XMAX,
                "ymin": 0,
                "ymax": YMAX,
            }
        )

    def set_fractures(self) -> None:
        # self._fractures = benchmark_2d_case_3(size=XMAX)
        points = np.array(
            [
                [[0.0500, 0.2200], [0.4160, 0.0624]],
                [[0.0500, 0.2500], [0.2750, 0.1350]],
                [[0.1500, 0.4500], [0.6300, 0.0900]],
                [[0.1500, 0.4000], [0.9167, 0.5000]],
                [[0.6500, 0.849723], [0.8333, 0.167625]],
                [[0.7000, 0.849723], [0.2350, 0.167625]],
                [[0.6000, 0.8500], [0.3800, 0.2675]],
                [[0.3500, 0.8000], [0.9714, 0.7143]],
                [[0.7500, 0.9500], [0.9574, 0.8155]],
                [[0.1500, 0.4000], [0.8363, 0.9727]],
            ]
        )
        xscale = XMAX / 2
        yscale = YMAX / 2
        points[:, 0] = xscale / 2 + points[:, 0] * xscale
        points[:, 1] = yscale / 2 + points[:, 1] * yscale
        self._fractures = [pp.LineFracture(pts) for pts in points]


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

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, sides.south, "dir")
        bc.internal_to_dirichlet(sd)
        return bc


class InitialCondition:
    def initial_condition(self) -> None:
        super().initial_condition()
        if self.params["setup"]["steady_state"]:
            num_cells = sum([sd.num_cells for sd in self.mdg.subdomains()])
            val = self.reference_variable_values.pressure * np.ones(num_cells)
            for time_step_index in self.time_step_indices:
                self.equation_system.set_variable_values(
                    val,
                    variables=[self.pressure_variable],
                    time_step_index=time_step_index,
                )

            for iterate_index in self.iterate_indices:
                self.equation_system.set_variable_values(
                    val,
                    variables=[self.pressure_variable],
                    iterate_index=iterate_index,
                )

            val = self.reference_variable_values.temperature * np.ones(num_cells)
            for time_step_index in self.time_step_indices:
                self.equation_system.set_variable_values(
                    val,
                    variables=[self.temperature_variable],
                    time_step_index=time_step_index,
                )

            for iterate_index in self.iterate_indices:
                self.equation_system.set_variable_values(
                    val,
                    variables=[self.temperature_variable],
                    iterate_index=iterate_index,
                )
        else:
            initial_state = self.params["setup"]["initial_state"]
            if initial_state != "ignore":
                vals = np.load(initial_state)
                self.equation_system.set_variable_values(vals, time_step_index=0)
                self.equation_system.set_variable_values(vals, iterate_index=0)


class Source:
    def locate_source(self, subdomains):
        source_loc_x = XMAX * 0.5
        source_loc_y = YMAX * 0.5
        ambient = [sd for sd in subdomains if sd.dim == self.nd]
        fractures = [sd for sd in subdomains if sd.dim == self.nd - 1]
        lower = [sd for sd in subdomains if sd.dim <= self.nd - 2]

        x, y, z = np.concatenate([sd.cell_centers for sd in fractures], axis=1)
        source_loc = np.argmin((x - source_loc_x) ** 2 + (y - source_loc_y) ** 2)
        src_frac = np.zeros(x.size)
        src_frac[source_loc] = 1

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
        t_inj = 40
        if self.params["setup"].get("isothermal", False):
            t_inj = self.reference_variable_values.temperature - 273

        t_inj = (
            self.units.convert_units(273 + t_inj, "K")
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
        residual_norm = np.linalg.norm(residual)  # / np.linalg.norm(reference_residual)
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
    CubicLawPermeability,
    FTHM_Solver.IterativeSolverMixin,
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
        dt_init = 1e0
        end_time = 1e1
    else:
        biot = 0.47
        dt_init = 1e-3
        if setup["grid_refinement"] >= 33:
            dt_init = 1e-4  # Is this necessary?
        end_time = setup.get("end_time", 5e2)
    porosity = 1.3e-2  # probably on the low side

    thermal_conductivity_multiplier = setup.get("thermal_conductivity_multiplier", 1)
    friction_coef = setup.get("friction_coef", 0.577)

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
                dilation_angle=5 * np.pi / 180,  # [rad]
                normal_permeability=1e-4,
                # granite
                biot_coefficient=biot,  # [-]
                density=2683.0,  # [kg * m^-3]
                porosity=porosity,  # [-]
                friction_coefficient=friction_coef,  # [-]
                # Thermal
                specific_heat_capacity=720.7,
                thermal_conductivity=0.1
                * thermal_conductivity_multiplier,  # Diffusion coefficient
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
            # iter_max=30,
            constant_dt=False,
        ),
        "units": pp.Units(kg=1e10),
        "meshing_arguments": {
            "cell_size": (0.1 * XMAX / cell_size_multiplier),
        },
        # experimental
        "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
        "linear_solver": {"preconditioner_factory": FTHM_Solver.thm_factory},
        # "linear_solver": "pypardiso",
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
            "nl_convergence_tol_res": 1e-7,
            "nl_divergence_tol": 1e8,
            "max_iterations": 10,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 1,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 0,  # Set to 0 to use turn off the tailored line search
        },
    )

    # write_dofs_info(model)
    # print(model.simulation_name())
    return model


if __name__ == "__main__":
    common_params = {
        "solver": "CPR",
    }
    for g in [
        1,
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
