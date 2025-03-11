"""This defines a 3d poromechanics model with 4 fractures. The setup is (slightly)
modified from the experiment reported in

    An efficient preconditioner for mixed-dimensional contact poromechanics based on the
    fixed stress splitting scheme, by Zabegaev, et al., arXiv:2501.07441.

The runscript is based on one found in

    https://github.com/Yuriyzabegaev/FTHM-Solver

and has been developed by Yuriy Zabegaev.



"""

import numpy as np
import porepy as pp
from porepy.numerics.nonlinear import line_search
from stats import SolverStatistics
from FTHM_Solver.hm_solver import IterativeHMSolver as Solver

XMAX = 1000
YMAX = 1000
ZMAX = 1000


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


class ConstraintLineSearchNonlinearSolver(
    line_search.ConstraintLineSearch,  # The tailoring to contact constraints.
    line_search.SplineInterpolationLineSearch,  # Technical implementation of the actual search along given update direction
    line_search.LineSearchNewtonSolver,  # General line search.
):
    """Collect all the line search methods in one class."""


class SolutionStrategyPM:
    def simulation_name(self) -> str:
        return "poromechanics_1"


class PoromechanicsModel(
    Geometry,
    BoundaryConditions,
    InitialCondition,
    Solver,
    SolverStatistics,
    pp.models.solution_strategy.ContactIndicators,
    pp.Poromechanics,
):
    pass


def make_model(setup: dict):
    cell_size_multiplier = setup["grid_refinement"]

    WEEK = 24 * 60 * 60 * 7

    shear = 1.2e10
    lame = 1.2e10
    biot = 0.47
    porosity = 1.3e-2
    specific_storage = 1 / (lame + 2 / 3 * shear) * (biot - porosity) * (1 - biot)

    params = {
        "setup": setup,
        "material_constants": {
            "solid": pp.SolidConstants(
                shear_modulus=shear,  # [Pa]
                lame_lambda=lame,  # [Pa]
                dilation_angle=0 * np.pi / 180,  # [rad]
                residual_aperture=1e-4,  # [m]
                normal_permeability=1e-4,
                permeability=1e-14,  # [m^2]
                # granite
                biot_coefficient=biot,  # [-]
                density=2683.0,  # [kg * m^-3]
                porosity=porosity,  # [-]
                specific_storage=specific_storage,  # [Pa^-1]
                # **get_barton_bandis_config(setup),
                # **get_friction_coef_config(setup),
            ),
            "fluid": pp.FluidComponent(
                compressibility=4.559 * 1e-10,  # [Pa^-1], fluid compressibility
                density=998.2,  # [kg m^-3]
                viscosity=1.002e-3,  # [Pa s], absolute viscosity
            ),
            "numerical": pp.NumericalConstants(
                characteristic_displacement=1e-1,  # [m]
            ),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            pressure=1e6,  # [Pa]
        ),
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=0.1 * WEEK,
            schedule=[0, 0.1 * WEEK],
            iter_max=25,
            constant_dt=True,
        ),
        "units": pp.Units(kg=1e10),
        "meshing_arguments": {
            "cell_size": (0.1 * XMAX / cell_size_multiplier),
        },
        # experimental
        "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
    }
    return PoromechanicsModel(params)


def run_model(setup: dict):
    model = make_model(setup)
    model.prepare_simulation()
    # print(model.simulation_name())
    # pp.plot_grid(model.mdg, plot_2d=True, fracturewidth_1d=5)

    print("Model grid:")
    print(model.mdg)
    print("\n")

    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "progressbars": False,
            "nl_convergence_tol": float("inf"),
            "nl_convergence_tol_res": 1e-7,
            "nl_divergence_tol": 1e8,
            "max_iterations": 100,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 1,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 1,  # Set to 0 to use turn off the tailored line search
        },
    )

    # print(model.simulation_name())


# if __name__ == "__main__":
if True:
    solver = 21
    for g in [0.25]:
        run_model(
            {
                "physics": 1,
                "geometry": 0.3,
                "barton_bandis_stiffness_type": 2,
                "friction_type": 1,
                "grid_refinement": g,
                "solver": solver,
                "permeability": 0,
            }
        )

    # solver = 2
    # for g in [0.25, 0.5, 1, 2, 3, 3.6]:
    #     run_model(
    #         {
    #             "physics": 1,
    #             "geometry": 0.3,
    #             "barton_bandis_stiffness_type": 2,
    #             "friction_type": 1,
    #             "grid_refinement": g,
    #             "solver": solver,
    #             "permeability": 0,
    #             # "save_matrix": True,
    #         }
    #     )
