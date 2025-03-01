# PACKAGES
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
import scipy.interpolate as spi
from multiprocessing import Pool
from scipy.interpolate import CubicSpline
from loguru import logger

# LOCAL FUNCTIONS
import modules.dynamics.system_eqs as system_eqs


def FI(i, ni, Xi_f, t_sub, GP, flag_integration):
    """
    Performs forward integration within single integration time frame, bounded by initial and final condition.

    :param i: integration frame index
    :param ni: number of integration frames
    :param Xi_f: initial condition of states for forward integration, in time instants x states
    :param t_sub: time-resolved time vector from initial to final instant of integration frame
    :param GP: dictionary containing matrix of coefficients Chi, linear and quadratic coeffs L,Q, among others
    :param flag_integration: flag indicating if integration is carried out in matrix form ('matrix') or einsum ('einsum')

    :return X_f, t_f: integrated states within t_f
    """

    # Parameters
    nx = len(Xi_f)
    C, L, Q = GP['C'], GP['L'], GP['Q']
    Chi = GP['Chi']

    # Time vectors for the integration
    t_f = np.copy(t_sub)
    t_span_f = [t_f[0], t_f[-1]]

    # Separate forward and backward integrations

    if flag_integration == 'matrix':
        sol_f = solve_ivp(lambda t, y: system_eqs.odeGalerkin_matrix(t, y, Chi), t_span_f, Xi_f, method='RK45',
                          t_eval=t_f)

    elif flag_integration == 'einsum':
        sol_f = solve_ivp(lambda t, y: system_eqs.odeGalerkin_einsum(t, y, C, L, Q), t_span_f, Xi_f, method='RK45',
                          t_eval=t_f)


    # Retrieve solutions
    t_f_sol = sol_f.t
    X_f = sol_f.y.T

    # If integration diverges and stops before, set states as null values
    if len(t_f_sol) < len(t_f):
        X_f = np.append(X_f, np.zeros((len(t_f) - len(t_f_sol), nx)), axis=0)
        t_f_sol = t_f

    logger.debug("Completed integration of " + str(i + 1) + "/" + str(ni - 1) + " time frames")

    return X_f, t_f

def BFI(i, ni, Xi_f, Xi_b, t_sub, GP, flag_integration):
    """
    Performs backward-forward weighted integration within single integration time frame, bounded by initial and final condition.

    :param i: integration frame index
    :param ni: number of integration frames
    :param Xi_f: initial condition of states for forward integration, in time instants x states
    :param Xi_b: initial condition on states for backward integration, in tim instants x stated
    :param t_sub: time-resolved time vector from initial to final instant of integration frame
    :param GP: dictionary containing matrix of coefficients Chi, linear and quadratic coeffs L,Q, among others
    :param flag_integration: flag indicating if integration is carried out in matrix form ('matrix') or einsum ('einsum')

    :return X_bf, t_bf: integrated states within t_bf
    """

    # Parameters
    nx = len(Xi_f)
    C, L, Q = GP['C'], GP['L'], GP['Q']
    Chi = GP['Chi']

    # Time vectors for the integration
    t_f = np.copy(t_sub)
    t_b = t_f[::-1]

    t_span_f = [t_f[0], t_f[-1]]
    t_span_b = [t_b[0], t_b[-1]]

    # Separate forward and backward integrations

    if flag_integration == 'matrix':
        sol_f = solve_ivp(lambda t,y: system_eqs.odeGalerkin_matrix(t,y,Chi), t_span_f, Xi_f, method='RK45', t_eval=t_f)
        sol_b = solve_ivp(lambda t,y: system_eqs.odeGalerkin_matrix(t,y,Chi), t_span_b, Xi_b, method='RK45', t_eval=t_b)

    elif flag_integration == 'einsum':
        sol_f = solve_ivp(lambda t,y: system_eqs.odeGalerkin_einsum(t,y,C,L,Q), t_span_f, Xi_f, method='RK45', t_eval=t_f)
        sol_b = solve_ivp(lambda t,y: system_eqs.odeGalerkin_einsum(t,y,C,L,Q), t_span_b, Xi_b, method='RK45', t_eval=t_b)

    # Retrieve solutions
    t_f_sol = sol_f.t
    X_f = sol_f.y.T
    t_b_sol = sol_b.t
    X_b = sol_b.y.T

    # If integration diverges and stops before, set states as null values
    if len(t_f_sol) < len(t_f):
        X_f = np.append(X_f, np.zeros((len(t_f) - len(t_f_sol), nx)), axis=0)
        t_f_sol = t_f
    if len(t_b_sol) < len(t_b):
        X_b = np.append(X_b, np.zeros((len(t_b) - len(t_b_sol), nx)), axis=0)
        t_b_sol = t_b

    # Weighting parameter
    nt = len(t_f)
    lim = 10
    weight = np.linspace(-lim, lim, nt - 2)
    sig = (1 / (1 + np.e ** (-weight))).reshape(-1, 1)

    # # Weighting alternative
    # xi = [0, (nt-1) / 2, nt-1]
    # yi = [0, 0.5, 1]
    # cs = spi.CubicSpline(xi, yi)
    # sig = (1 - cs(np.arange(0, nt))).reshape(-1, 1)
    # sig = sig[1:-1]

    # Weight solutions
    X_b = np.flip(X_b, axis=0)

    X_bf = np.zeros((nt, nx))
    X_bf[0, :] = X_f[0, :]
    X_bf[-1, :] = X_b[-1, :]

    X_weighted = np.multiply(X_f[1:-1, :], (1 - sig)) + np.multiply(X_b[1:-1, :], sig)
    X_bf[1:-1, :] = X_weighted

    logger.debug("Completed integration of " + str(i+1) + "/" + str(ni-1) + " time frames")

    return X_bf, t_f

def integrator(Xi, ti, Dt, GP, Phi, flag_integration, N_process):
    """
    Performs a physically-informed integration in between available snapshots. Leverages the dynamical system retrieved
    from Galerkin Proejections

    :param Xi: initial conditions of states, corresponding to POD temporal coefficients projected from available NTR snapshots
    :param ti: time instants corresponding to initial conditions
    :param Dt: time separation objective (time-resolved)
    :param Phi: spatial truncated POD modes
    :param GP: dictionary containing matrix of coefficients Chi, linear and quadratic coeffs L,Q, among others
    :param flag_integration: flag indicating if integration is carried out in matrix form ('matrix') or einsum ('einsum')
    :param N_process: number of cores used for parallel integration

    :return test_GP: dictionary containing states of the system with temporal resolution, as well as snapshot matrix and time vector
    """

    # Parameters
    tol = 1e-5
    ni = len(ti)
    nx = np.shape(Xi)[1]

    # Create time-resolved time vector
    t = np.array([ti[0]])
    for i in range(ni-1):
        t_sub = np.arange(ti[i], ti[i+1] + Dt*tol, Dt)
        t = np.concatenate((t, t_sub[1:]))
    nt = len(t)

    # Initialize time-resolved coordinates
    X = np.zeros((nt, nx))
    X[0, :] = Xi[0, :]

    # Non-parallelized integration
    if N_process == 1:
        logger.info("Non-parallel integration")
        for i in range(ni-1):

            t_bf = np.arange(ti[i], ti[i + 1] + Dt * tol, Dt)
            X_bf = BFI(i, ni, Xi[i,:], Xi[i+1,:], t_bf, GP, flag_integration)[0]

            it0 = np.where(np.abs(ti[i] - t) < tol)[0][0]
            itf = np.where(np.abs(ti[i+1] - t) < tol)[0][0]
            X[it0+1:itf+1, :] = X_bf[1:, :]


    # Parallelized integration with multiple processes
    else:
        logger.info("Parallel integration")
        with Pool(processes=N_process) as pool:
            # issue multiple tasks each with multiple arguments
            results = [pool.apply_async(BFI, args=(i, ni, Xi[i,:], Xi[i+1,:],
                       np.arange(ti[i], ti[i + 1] + Dt * tol, Dt), GP,
                       flag_integration)).get() for i in range(ni-1)]

        for result in results:
            X_bf, t_bf = result

            it0 = np.where(np.abs(t_bf[0] - t) < tol)[0][0]
            itf = np.where(np.abs(t_bf[-1] - t) < tol)[0][0]
            X[it0 + 1:itf+1, :] = X_bf[1:, :]

    Ddt = np.dot(Phi, X.T)
    test_GP = {'t': t, 'X': X, 'Ddt': Ddt}

    return test_GP

def interpolator(Xi, ti, Dt, Phi):
    """
    Interpolates states of the system in between available ICs using a cubic spline function

    :param Xi: initial conditions of states, corresponding to POD temporal coefficients projected from available NTR snapshots
    :param ti: time instants corresponding to initial conditions
    :param Dt: time separation objective (time-resolved)
    :param Phi: spatial truncated POD modes

    :return test_interp: dictionary containing states of the system with temporal resolution, as well as snapshot matrix and time vector
    """

    # Parameters
    tol = 1e-3
    ni = len(ti)
    nx = np.shape(Xi)[1]

    # Create time-resolved time vector
    t = np.array([ti[0]])
    for i in range(ni-1):
        t_sub = np.arange(ti[i], ti[i+1] + Dt*tol, Dt)
        t = np.concatenate((t, t_sub[1:]))
    nt = len(t)

    # Initialize time-resolved coordinates
    X = np.zeros((nt, nx))

    # Spline interpolation process
    for i in range(nx):
        X[:,i] = CubicSpline(ti, Xi[:,i])(t)

    Ddt = np.dot(Phi, X.T)
    test_interp = {'t': t, 'X': X, 'Ddt': Ddt}

    return test_interp


def get_TH(grid, Ddti, D_mean, ti, Dt):
    """
    Propagates velocity fields backward and forward in time (interpolated/integration process). Follows procedure of
    Scarano&Moore(2012) in "An advection-based model to increase the temporal resolution of PIV time series"
    with some modifications. Valid only for available horizontal and vertical velocity fields.

    :param grid: structure containing grid information
    :param Ddti: fluctuating velocity fields with no temporal resolution
    :param D_mean: mean velocity field
    :param ti: time vector with no temporal resolution
    :param Dt: time separation objective (time-resolved)

    :return: structure containing states of the system with temporal resolution
    """

    # Parameters
    X = grid['X']
    Y = grid['Y']
    M = grid['B']

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    xmin = np.min(X)
    xmax = np.max(X)
    ymin = np.min(Y)
    ymax = np.max(Y)

    # Initial conditions
    ni = len(ti)

    Di = Ddti + D_mean
    Ui = np.reshape(Di[0:m * n, :], (m, n, ni), order='F')
    Vi = np.reshape(Di[m * n: 2 * m * n, :], (m, n, ni), order='F')

    # Start propagation/integration process
    c = 0
    t = []
    U, V = np.empty((2, m, n, 1))
    for i in range(ni - 1):
        ts = ti[i + 1] - ti[i]
        t_sub = np.arange(ti[i], ti[i + 1] + Dt, Dt)  # Temporal vector with time resolution in between ICs

        Uc1 = gaussian_filter(Ui[:, :, i],
                              7)  # Advection velocity for time instant i corresponding to horizontal velocity. A local filter
        # is applied each space in between ICs to get the largest fluctuations, responsible for advection
        Uc3 = gaussian_filter(Ui[:, :, i + 1],
                              7)  # Advection velocity for time instant i+1 corresponding to horizontal velocity

        Vc1 = gaussian_filter(Vi[:, :, i],
                              7)  # Advection velocity for time instant i corresponding to vertical velocity
        Vc3 = gaussian_filter(Vi[:, :, i + 1],
                              7)  # Advection velocity for time instant i+1 corresponding to vertical velocity

        for j in range(len(t_sub) - 1):
            c = c + 1

            # Points in the grid where the values (X, Y) have propagated
            X1 = X + (t_sub[0] - t_sub[j]) * Uc1
            Y1 = Y + (t_sub[0] - t_sub[j]) * Vc1

            X3 = X + (t_sub[-1] - t_sub[j]) * Uc3
            Y3 = Y + (t_sub[-1] - t_sub[j]) * Vc3

            # If propagated points are out of limits, restrict them to the limits
            X1[X1 < xmin] = xmin
            X1[X1 > xmax] = xmax
            Y1[Y1 < ymin] = ymin
            Y1[Y1 > ymax] = ymax

            X3[X3 < xmin] = xmin
            X3[X3 > xmax] = xmax
            Y3[Y3 < ymin] = ymin
            Y3[Y3 > ymax] = ymax

            # Interpolate velocity at propagated points at time instants i or i+1. Maintain region of mask as available time instants
            U1 = spi.RegularGridInterpolator((X[0, :].T, np.flip(Y[:, 0])), Ui[:, :, i], method='cubic', fill_value=0)(
                X1[0, :].T, np.flip(Y[:, 0]))
            U1 = U1 + np.dot((Ui[:, :, i] - U1), M)

            U3 = spi.RegularGridInterpolator((X[0, :].T, np.flip(Y[:, 0])), Ui[:, :, i + 1], method='cubic',
                                             fill_value=0)(X3[0, :].T, np.flip(Y[:, 0]))
            U3 = U3 + np.dot((Ui[:, :, i + 1] - U3), M)

            V1 = spi.RegularGridInterpolator((X[0, :].T, np.flip(Y[:, 0])), Vi[:, :, i], method='cubic', fill_value=0)(
                X1[0, :].T, np.flip(Y[:, 0]))
            V1 = V1 + np.dot((Vi[:, :, i] - V1), M)

            V3 = spi.RegularGridInterpolator((X[0, :].T, np.flip(Y[:, 0])), Vi[:, :, i + 1], method='cubic',
                                             fill_value=0)(X3[0, :].T, np.flip(Y[:, 0]))
            V3 = V3 + np.dot((Vi[:, :, i + 1] - V3), M)

            # Weighted forward and backward predictions
            U = np.concatenate((U, (t_sub[-1] - t_sub[j]) / ts * U1 + (t_sub[j] - t_sub[0]) / ts * U3), axis=2)
            V = np.concatenate((V, (t_sub[-1] - t_sub[j]) / ts * V1 + (t_sub[j] - t_sub[0]) / ts * V3), axis=2)

            if c == 1:
                TH = {'U': U.reshape((m * n, 1)), 'V': V.reshape((m * n, 1)), 't': t_sub[j].reshape(1, 1)}
            else:
                TH['U'] = np.append(TH['U'], U.reshape((m * n, 1)), axis=1)
                TH['V'] = np.append(TH['V'], V.reshape((m * n, 1)), axis=1)
                TH['t'] = np.append(TH['t'], t_sub[j].reshape(1, 1), axis=1)
        t.append(t_sub)

    # Final values
    nt = np.shape(U)[2]
    t = np.array(t).reshape(-1, 1)
    U, V = np.reshape(U, (m * n, nt), order='F'), np.reshape(V, (m * n, nt), order='F')
    Ddt = np.concatenate((U, V), axis=0) - D_mean
    test_TH = {'t': t, 'Ddt': Ddt}

    return test_TH