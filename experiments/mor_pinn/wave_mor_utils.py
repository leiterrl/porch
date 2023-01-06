'''Wave RBM by Glas, Patera, Urban'''
import os
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt
import pickle

from pymor.analyticalproblems.functions import ExpressionFunction
from pymor.discretizers.builtin.cg import InterpolationOperator, CGVectorSpace
from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.gui.visualizers import OnedVisualizer
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import LincombOperator, IdentityOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Parameter, ParameterType
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.algorithms.greedy import rb_greedy
from pymor.algorithms.timestepping import TimeStepper
from pymor.reductors.wave_so import WaveRBReductor
from pymor.operators.constructions import induced_norm
from pymor.vectorarrays.numpy import NumpyVectorArray

class WaveEquationModel(InstationaryModel):
    def __init__(self, T, rhs, time_stepper=None, num_values=None, output_functional=None, mass=None, operator=None,
                 initial_time=0, initial_data=None, initial_velocity=None, products=None, parameter_space=None, estimator=None, visualizer=None, name=None):
        '''Linear wave equation on [-1, 1].'''
        # assert isinstance(time_stepper, WaveEquationTimeStepper)

        super().__init__(
            T,
            initial_data,
            operator,
            rhs,
            mass=mass,
            time_stepper=time_stepper,
            num_values=num_values,
            output_functional=output_functional,
            products=products,
            estimator=estimator,
            visualizer=visualizer,
            name=name,
            parameter_space=parameter_space
        )
        self.initial_velocity = initial_velocity
        if initial_time is None:
            initial_time = 0.
        self.initial_time = initial_time

    
    def _solve(self, mu=None, return_output=False):
        mu = self.parse_parameter(mu).copy()

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            self.logger.info(f'Solving {self.name} for {mu} ...')

        mu['_t'] = self.initial_time
        U0 = self.initial_data.as_range_array(mu)
        V0 = self.initial_velocity.as_range_array(mu)
        U = self.time_stepper.solve(operator=self.operator, rhs=self.rhs, initial_data=U0, initial_velocity=V0,
                                    mass=self.mass, initial_time=self.initial_time, end_time=self.T, mu=mu, num_values=self.num_values)
        if return_output:
            if self.output_functional is None:
                raise ValueError('Model has no output')
            return U, self.output_functional.apply(U, mu=mu)
        else:
            return U


class OnedScipyWaveEquationModel(WaveEquationModel):
    def __init__(self, T, rhs, time_stepper=None, initial_time=None, initial_data=None, initial_velocity=None, grid=None,
                 num_values=None, output_functional=None, estimator=None, parameter_space=None, visualizer=None):

        self.grid = grid
        num_intervals = grid.num_intervals
        dx = (grid.domain[1] - grid.domain[0]) / num_intervals

        if visualizer is None:
            visualizer = OnedVisualizer(self.grid)
        so_space = initial_data.range

        # 1D FEM with Piece-wise linear Ansatz functions and hom. Dirichlet
        # modification in comparison to Glas et al: included FEM mass matrix
        self.l2_product = NumpyMatrixOperator(
            dx/6 * diags(
                [
                    4 * np.ones(num_intervals+1),
                    np.ones(grid.num_intervals),
                    np.ones(grid.num_intervals)
                ],
                [0, 1, -1],
                format='csc'
            ),
            source_id=so_space.id,
            range_id=so_space.id
        )
        self.h1_product = NumpyMatrixOperator(
            -1/dx * diags(
                [
                    -2 * np.ones(grid.num_intervals+1),
                    np.ones(grid.num_intervals),
                    np.ones(grid.num_intervals)
                ],
                [0, 1, -1],
                format='csc'
            ),
            source_id=so_space.id,
            range_id=so_space.id
        )
        operator = LincombOperator(
            [NumpyMatrixOperator(
                1/dx * diags(
                    [
                        -2 * np.ones(grid.num_intervals+1),
                        np.ones(grid.num_intervals),
                        np.ones(grid.num_intervals)
                    ],
                    [0, 1, -1],
                    format='csc'
                ),
                source_id=so_space.id,
                range_id=so_space.id
            )],
            [ExpressionParameterFunctional('-wave_speed**2', {'wave_speed': 0.})]
        )
        mass = self.l2_product

        super().__init__(
            T,
            rhs=rhs,
            time_stepper=time_stepper,
            num_values=num_values,
            output_functional=output_functional,
            initial_time=initial_time,
            initial_data=initial_data,
            initial_velocity=initial_velocity,
            products={'l2': self.l2_product},
            estimator=estimator,
            visualizer=visualizer,
            mass=mass,
            operator=operator,
            parameter_space=parameter_space,
            name='wave_equation_1d'
        )


class TravellingWaveEquationModel(OnedScipyWaveEquationModel):
    def __init__(self, T, num_intervals, initial_time=None, initial_velocity_type='zero', time_stepper=None, num_values=None,
                 output_functional=None, estimator=None, visualizer=None, q0_supp=1.5):
        assert initial_velocity_type in ['zero', 'travelling']
        self.initial_velocity_type = initial_velocity_type
        self.q0_supp = q0_supp

        grid = OnedGrid(domain=(-1, 1), num_intervals=num_intervals)
        self.num_intervals = num_intervals

        initial_data = InterpolationOperator(
            grid,
            ExpressionFunction('(x >= -q0_supp/2) * (x <= q0_supp/2) * 1/2 * (1 + cos(2*pi/q0_supp * x))', \
                               parameter_type={'q0_supp': ()}),
        )
        initial_data = VectorOperator(initial_data.as_vector({'q0_supp': q0_supp}))

        if initial_velocity_type == 'zero':
            initial_velocity = VectorOperator(initial_data.range.zeros())

        elif initial_velocity_type == 'travelling':
            # travelling wave: v0 = (+/-1) * wave_speed * d(u0)/dx
            initial_velocity = InterpolationOperator(
                grid,
                ExpressionFunction('(x >= -q0_supp/2) * (x <= q0_supp/2) * pi/q0_supp * sin(2*pi/q0_supp * x)', \
                                   parameter_type={'q0_supp': ()}),
            )
            initial_velocity = LincombOperator(
                [VectorOperator(initial_velocity.as_vector({'q0_supp': q0_supp}))],
                [ExpressionParameterFunctional('-wave_speed', {'wave_speed': 0.})]
            )
        else:
            raise ValueError('Unknown initial_velocity_type: ' + initial_velocity_type)

        super().__init__(
            T, 
            rhs=None,
            grid=grid,
            time_stepper=time_stepper,
            output_functional=output_functional,
            initial_time=initial_time,
            initial_data=initial_data,
            initial_velocity=initial_velocity,
            estimator=estimator,
            visualizer=visualizer,
            parameter_space=CubicParameterSpace(
                ParameterType({'wave_speed': 0.}),
                ranges={'wave_speed': (-2, 2)}
            ),
        )

class GlasEtAlWaveEquationModel(OnedScipyWaveEquationModel):
    def __init__(self, T, num_intervals, initial_time=None, time_stepper=None, num_values=None, output_functional=None,
                 estimator=None, visualizer=None):

        grid = OnedGrid(domain=(0, 1), num_intervals=num_intervals)
        self.num_intervals = num_intervals
        dx = (grid.domain[1] - grid.domain[0]) / num_intervals

        space = CGVectorSpace(grid)

        initial_data = VectorOperator(space.zeros())
        initial_velocity = VectorOperator(space.zeros())
        first_node_dirichlet_mass = space.make_array(np.hstack([1, np.zeros(space.dim-1)]))
        first_node_dirichlet_stiffness = first_node_dirichlet_mass.copy()
        first_node_dirichlet_mass.scal(dx/6)
        first_node_dirichlet_stiffness.scal(1/dx)
        rhs = LincombOperator(
            [VectorOperator(first_node_dirichlet_mass),
             VectorOperator(first_node_dirichlet_stiffness)],
            [ExpressionParameterFunctional('150/(cosh(5*_t)**2) * tanh(5*_t) * (1/(cosh(5*_t)**2) - tanh(5*_t)**2)', {'_t': 0.}),
             ExpressionParameterFunctional('wave_speed**2 * tanh(5*_t)**3', {'_t': 0., 'wave_speed': 0.})]
        )

        super().__init__(
            T,
            rhs=rhs,
            grid=grid,
            time_stepper=time_stepper,
            output_functional=output_functional,
            initial_time=initial_time,
            initial_data=initial_data,
            initial_velocity=initial_velocity,
            estimator=estimator,
            visualizer=visualizer,
            parameter_space=CubicParameterSpace(
                ParameterType({'wave_speed': 0.}),
                ranges={'wave_speed': (0.3, 2)}
            ),
        )


class WaveEquationTimeStepper(TimeStepper):
    '''Implicit-Explicit time-stepping scheme used in Glas et al.'''
    def __init__(self, nt, theta=.25):
        self.__auto_init(locals())

    def solve(self, initial_time, end_time, initial_data, initial_velocity, operator, rhs=None, mass=None, mu=None, num_values=None):
        assert num_values is None, 'num_values not implemented'

        t0 = initial_time
        t1 = end_time
        nt = self.nt
        dt = (t1 - t0) / nt
        t = t0 + dt
        A = operator
        b = rhs

        theta = self.theta

        so_space = A.range

        if b is None:
            b_time_dep = False
        elif isinstance(b, Operator):
            assert b.source.dim == 1
            assert b.range == A.range
            b_time_dep = b.parametric and '_t' in b.parameter_type
            if not b_time_dep:
                dtsq_gh = b.as_vector(mu) * dt**2
            else:
                mu['_t'] = t0
                g_old_old = b.as_vector(mu)
                mu['_t'] = t
                g_old = b.as_vector(mu)
        else:
            assert len(b) == 1
            assert b in A.range
            b_time_dep = False
            dtsq_gh = b * dt**2

        if mass is None:
            mass = NumpyMatrixOperator(
                eye(so_space.dim),
                source_id=so_space.id,
                range_id=so_space.id
            )

        L_I = mass + theta * dt**2 * A
        L_E_1 = 2 * mass - (1 - 2*theta) * dt**2 * A
        L_E_2 = -(mass + theta * dt**2 * A)

        assert not L_I.parametric or '_t' not in L_I.parameter_type
        assert not L_E_1.parametric or '_t' not in L_E_1.parameter_type
        L_E_1 = L_E_1.assemble(mu)
        L_E_2 = L_E_2.assemble(mu)
        L_I = L_I.assemble(mu)

        R = A.source.empty(reserve=nt)

        U = initial_data.copy() + dt * initial_velocity.copy()
        U_old = initial_data.copy()

        R.append(U_old)
        R.append(U)

        for _ in np.arange(2, nt):
            t += dt
            mu['_t'] = t
            rhs = L_E_1.apply(U) + L_E_2.apply(U_old)
            if b:
                if b_time_dep:
                    g = b.as_vector(mu)
                    dtsq_gh = (theta * g + (1-2*theta) * g_old + theta * g_old_old) * dt**2
                    g_old_old = g_old
                    g_old = g

                rhs += dtsq_gh


            U_old = U.copy()
            U = L_I.apply_inverse(rhs)
            R.append(U)

        return R


def compute_rom(fom, training_set, max_extensions=None, rtol=1e-2):
    '''Compute reduced-order model (ROM) for wave equation.
        fom
            Full-order model
        training_set
            finite subset of parameter space
        max_extensions
            maximal extensions in greedy algorithm
        rtol
            relative error tolerance in greedy algorith,
    '''
    fom.enable_caching('memory')

    reductor = WaveRBReductor(
        fom,
        coercivity_estimator=squared_wave_speed,
        continuity_estimator=squared_wave_speed
    )

    greedy_data = rb_greedy(
        fom,
        reductor,
        training_set,
        rtol=rtol,
        use_estimator=True,
        max_extensions=max_extensions
    )
    return greedy_data['rom'], reductor, greedy_data

def save_rom_data(rom, reductor, greedy_data, filepath):
    folder = os.path.dirname(filepath)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    f = open(filepath, 'wb')
    pickle.dump((rom, reductor, greedy_data), f)
    f.close()

def load_rom_data(filepath):
    f = open(filepath, 'rb')
    rom, reductor, greedy_data = pickle.load(f)
    f.close()
    return rom, reductor, greedy_data

def eval_rom(old_rom, reductor, mu, initial_time, final_time, n_time_steps, initial_data, initial_velocity):
    if isinstance(initial_data, NumpyVectorArray):
        initial_data = VectorOperator(initial_data)
    if isinstance(initial_velocity, NumpyVectorArray):
        initial_velocity = VectorOperator(initial_velocity)
    rom = reductor.update_initial_data(old_rom, initial_data, initial_velocity)
    updated_attributes = {'time_stepper':rom.time_stepper.with_(nt=n_time_steps), 'initial_time': initial_time, 'T':final_time}
    rom = rom.with_(**updated_attributes)
    u = rom.solve(mu)
    U_rom = reductor.reconstruct(u)
    err_est = rom.estimator.estimate(u, mu=mu, m=rom, return_error_sequence=True)
    return U_rom, err_est

def squared_wave_speed(mu):
    return mu['wave_speed']**2

def generate_fom(scenario):
    assert scenario in ('zero', 'travelling', 'GlasEtAl')
    if scenario in ('zero', 'travelling'):
        # problem in zero szenario: estimate does not hold for \dot{e} which it should
        T = 2
        n_spatial = np.int(1e3)
        n_time = 500
        # adjust support of q0 (the smaller the harder for RB)
        # should be between 0.5 and 2.0
        #   e.g. 0.1 is too small - will generate no good basis (with max_extensions=30) but already the fom is bad (artificial oscillations)
        #   maximal reasonable value is 2 since otherwise boundary conditions might be violated by intial value
        q0_supp = 0.5
        time_stepper = WaveEquationTimeStepper(n_time)
        fom = TravellingWaveEquationModel(T, n_spatial, initial_velocity_type=scenario, time_stepper=time_stepper, q0_supp=q0_supp)
        n_wave_speed = 4
        max_extensions = 10
        rtol = 1e-2
    elif scenario == 'GlasEtAl':
        # problems:
        #   experiments do not exactly match paper (but quite good right now)
        #   V_Err is not bounded by the estimator
        #       which is a problem since errors in the initial_velocity give bad efficiencies
        T = 1
        n_spatial = 400
        n_time = 100
        time_stepper = WaveEquationTimeStepper(n_time)
        fom = GlasEtAlWaveEquationModel(T, n_spatial, time_stepper=time_stepper)
        n_wave_speed = 60
        max_extensions = 33
        rtol=None #enforce 33 iterations in greedy to repeat experiments of the paper
    else:
        raise ValueError('Unkown scenario: {}'.format(scenario))

    return fom, n_wave_speed, max_extensions, rtol