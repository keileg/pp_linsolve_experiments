import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import porepy as pp
import numpy as np
import time
from porepy.numerics.nonlinear import line_search

from stats import SolverStatistics
from FTHM_Solver.thm_solver import THMSolver

XMAX = 1000
YMAX = 1000


class Geometry:
    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000, "zmin": 0, "zmax": 1000}
        )

    def set_fractures(self) -> None:
        # Set four fractures in the domain.
        pts_list = np.array(
            [
                [[0.1, 0.1, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5], [0.2, 0.8, 0.8, 0.2]],
                [[0.15, 0.15, 0.4, 0.4], [0.7, 0.7, 0.2, 0.2], [0.2, 0.8, 0.8, 0.2]],
                [[0.45, 0.45, 0.6, 0.6], [0.3, 0.3, 0.8, 0.8], [0.2, 0.8, 0.8, 0.2]],
                [[0.6, 0.6, 0.8, 0.8], [0.2, 0.2, 0.8, 0.8], [0.2, 0.8, 0.8, 0.2]],
            ]
        )
        box = self._domain.bounding_box
        pts_list[:, 0] *= box["xmax"]
        pts_list[:, 1] *= box["ymax"]
        pts_list[:, 2] *= box["zmax"]

        self._fractures = [pp.PlaneFracture(pts) for pts in pts_list]


class BoundaryConditions:
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # Dirichlet condition for the pressure on all sides
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_values_pressure(self, boundary_grid):
        # Default values everywhere but the east side, where the pressure is elevated
        # by a factor of 20.
        vals = super().bc_values_pressure(boundary_grid)
        sides = self.domain_boundary_sides(boundary_grid)
        val = 20
        vals[sides.east] *= val
        return vals

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, sides.south, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros((self.nd, boundary_grid.num_cells))

        val = self.units.convert_units(3e6, units="Pa")

        # South side has Dirichlet condition. Tensional force on the north side (?),
        # compressive force on the remaining sides.
        bc_values[0, sides.north] = 0.1 * val * boundary_grid.cell_volumes[sides.north]
        bc_values[2, sides.bottom] = val * boundary_grid.cell_volumes[sides.bottom]
        bc_values[0, sides.west] = val * boundary_grid.cell_volumes[sides.west]
        bc_values[2, sides.top] = -val * boundary_grid.cell_volumes[sides.top]
        bc_values[0, sides.east] = -val * boundary_grid.cell_volumes[sides.east]

        return bc_values.ravel("F")


class InitialCondition:
    def initial_condition(self) -> None:
        # Set initial condition for pressure, default values for other variables.
        super().initial_condition()
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


class Source:
    def locate_source(self, subdomains):
        source_loc_x = XMAX * 0.9
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
        t_inj = (
            self.units.convert_units(273 + 40, "K")
            - self.reference_variable_values.temperature
        )
        src *= cv * t_inj
        return super().energy_source(subdomains) + pp.ad.DenseArray(src)


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
    THMSolver,
    SolverStatistics,
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
                dilation_angle=5 * np.pi / 180,  # [rad]
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
        "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
    }
    return THMModel(params)


def run_model(setup: dict):
    model = make_model(setup)
    model.prepare_simulation()
    print(model.simulation_name())

    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "progressbars": False,
            "nl_convergence_tol": float("inf"),
            "nl_convergence_tol_res": 1e-7,
            "nl_divergence_tol": 1e8,
            "max_iterations": 30,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 0,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 1,  # Set to 0 to use turn off the tailored line search
        },
    )

    # write_dofs_info(model)
    print(model.simulation_name())


if __name__ == "__main__":
    common_params = {
        "geometry": "4test",
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
