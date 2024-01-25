import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp, trapz
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pyqosc.shortcut import shortcut_metrics, linear_impens, hyperbolic_spiral
from pyqosc.general import qdistance_to_ss
import copy

class vdp:
    def __init__(self, N, omega_0 = 1, omega = 1, Omega_1 = 0, Omega_2 = 1, gamma_1 = 1, gamma_2 = 0.1):
        self.name = "vdP"
        self.N = N
        self.omega_0 = omega_0
        self.omega = omega
        self.Delta = self.omega - self.omega_0
        self.Omega_1 = Omega_1
        self.Omega_2 = Omega_2
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        
        self.Ham = None
        self.c_ops = None
    
    def dynamics(self, Liouvillian = False, b = None):
        
        if not(b):
            b = qt.destroy(self.N)
            
        cond_A = isinstance(self.Omega_1, (list, np.ndarray))
        cond_B = isinstance(self.Omega_2, (list, np.ndarray))
        if cond_A or cond_B:
            self.Ham = [-self.Delta * b.dag() * b]
            
            if not(cond_A):
                self.Ham[0] += (b+b.dag()) * self.Omega_1
            else:
                self.Ham.append([(b+b.dag()), self.Omega_1])
                
            if not(cond_B):
                self.Ham[0] += 1j * (b - b.dag()) * self.Omega_2
            else:
                self.Ham.append([1j*(b-b.dag()), self.Omega_2])
        else:
            self.Ham = -self.Delta * b.dag() * b + (b+b.dag()) * self.Omega_1 + 1j * (b - b.dag()) * self.Omega_2
            
        self.c_ops = [np.sqrt(self.gamma_1) * b.dag(), np.sqrt(self.gamma_2) * b**2]
    
        if Liouvillian:
            return qt.liouvillian(self.Ham, self.c_ops)
        else:
            return self.Ham, self.c_ops
            
    def couple(self, vdp2, D, Liouvillian = False, antiphase = False):
        b1 = qt.tensor(qt.destroy(self.N), qt.qeye(self.N))
        b2 = qt.tensor(qt.qeye(self.N), qt.destroy(self.N))
        
        Ham1, c_ops1 = self.dynamics(b = b1)
        Ham2, c_ops2 = vdp2.dynamics(b = b2)
        
        mul = -1
        if antiphase:
            mul = 1
        
        self.Ham = Ham1 + Ham2
        self.c_ops = c_ops1 + c_ops2 + [np.sqrt(D) * (b1 + mul * b2)]
        
        if Liouvillian:
            return qt.liouvillian(self.Ham, self.c_ops)
        else:
            return self.Ham, self.c_ops

    def evolve(self, rho_0, timelst, expect = [], plot = False, overlap_with = None, **plot_kwargs):
        '''
        ## INTRODUCTION
        
        Evolve the vdp oscillator system with the Hamiltonian and collapse operators given by ``vdp.dynamics``,
        using mesolve. 
        
        ---
        
        ## RETURNS
        
        Returns the resulting states. If ``expect`` is specified, then return a list of expectation values instead.
        
        ---
        
        ## PARAMETERS
        
        ``rho_0``   :
            Initial density matrix.
            
        ``timelst`` :
            Evolution time list.
            
        ``expect``  :
            A ``list`` of operators to take the expectation values of.
        
        ``plot``    :
            Plot the results. If the expectation value is complex or imaginary, then the plot is a parametric 
            curve with the time as the parameter. If the expectation value is real, then the plot is a plot of
            the expectation value versus time. 
        
        ``overlap_with``    :   ``None``
            ``matplotlib.axes.Axes`` object to plot on. If not specified, then make a new figure and axis with figure
            size ``figsize=(5,5)``.
            
        ``**plot_kwargs``   :
            Optional keyword arguments for the ``matplotlib.axes.Axes.plot`` command.
        
        '''
        
        if not(self.Ham):
            self.dynamics()
        
        res = qt.mesolve(self.Ham, rho_0, timelst, self.c_ops, expect, options = qt.Options(nsteps = int(1e8)))
        
        if expect:
            out = res.expect
        else:
            out = res.states
        
        if plot and expect:
            if not(overlap_with):
                fig, ax = plt.subplots(1, figsize = (5, 5))
            else:
                ax = overlap_with
            
            for i in range(len(expect)):
                val = out[i]
                if np.complex128 in [type(x) for x in val]:
                    ax.plot(np.real(val), np.imag(val), label = f"expect_{i}",**plot_kwargs)
                    ax.scatter(np.real(val[0]), np.imag(val[0]), color = "r", label = "start")
                    ax.scatter(np.real(val[-1]), np.imag(val[-1]), color = "g", label = "finish")
                else:
                    ax.plot(timelst, val, label = f"expect_{i}", **plot_kwargs)
             
        return out

    def adler(self, t_end = 1e2, t_eval = 10, timepoints_returned = 100, method = "Radau",
                init_polar = [1, 0], rho_0 = None, plot = False, overlap_with = None, color = "k",
                one_cycle = False, post_multiplier = np.sqrt(2)):
        '''
        Solve the equation of motion for the expectation value of the annihilation operator
        (https://doi.org/10.1103/PhysRevLett.112.094102).
        
        This method does NOT work with time-dependent ``Omega_i``. If ``Omega_i`` is time 
        dependent, then only the first entry is taken as the value for evaluation. 
        
        ----------
        Returns
        ----------
        A tuple containing the time list, ``r``, ``phi``, and ``beta = r*exp(1j*phi)``, in that order, for the chosen
        evaluation interval.
        
        ----------
        Parameters
        ----------
        
        ``t_end``       : ``100``
            End of solution computation. 
            
        ``t_eval``      : ``10``
            How far from ``t_end`` the result is returned, i.e. the function returns the result from ``t_end-t_eval`` until ``t_end``.
        
        ``timepoints_returned`` : ``100``
            Number of timepoints returned.

        ``method``  : ``Radau``
            Method for ``scipy.integrate.solve_ivp`` evaluation.
            
        ``init_polar``  : ``[1,0]``
            Initial value for ``r`` and ``phi``, respectively. ``r=0`` is an invalid initial condition.
            
        ``rho_0``   : ``None``
            Initial density matrix of the oscillator. If this is input, then ``init_polar`` is overriden by the value corresponding to
            this density matrix. 
        
        ``plot``    : ``False``
            Put in ``x`` and ``y`` obtained from ``r`` and  ``phi`` for plotting.
            
        ``overlap_with``    : None
            Plot in the given axis. If ``None``, then make a new matplotlib figure and axes object.
            
        ``color``   :   ``"k"``
            Color of the curve.
            
        ``one_cycle``   :   ``False``
            Return only one cycle of the motion evaluated from ``t_end-t_eval``.
            
        ``post_multiplier``   :   ``numpy.sqrt(2)``
            Multiplies the modulus of the result. The definition of the phase space coordinate may be different. 
            The original equation of motion determines the evolution of ``b = 2**(-0.5) * (x + 1j*p)``. The default
            value of ``sqrt(2)`` is used to match the result with that of (https://doi.org/10.1103/PhysRevLett.112.094102)
            which uses ``x/x_zpf`` and ``p/p_zpf`` as the phase plane coordinate.
            
        '''
        
        t_return = np.linspace(t_end-t_eval, t_end, timepoints_returned)
        
        def adler_eq(t, y):
            r, phi = y
            return [(self.gamma_1/2-self.gamma_2*r**2)*r-self.Omega_1*np.sin(phi)-self.Omega_2*np.cos(phi), 
                    self.Delta - self.Omega_1/r*np.cos(phi) + self.Omega_2/r*np.sin(phi)]             
                
        sol = solve_ivp(adler_eq, t_span = [0, t_end], y0 = init_polar, dense_output=True, method = method, rtol = 1e-9)
        sol_vals = sol.sol(t_return)
        
        r = sol_vals[0] * np.sqrt(2) 
        phi = sol_vals[1]
        
        for i in range(timepoints_returned):
            if r[i]<0:
                r[i] *= -1
                phi[i] += np.pi
            phi[i] = phi[i]%(2*np.pi)
        
        if one_cycle:
            r, phi = self._get_cycle(r, phi)
        
        beta = r * np.exp(1j * phi)

        if plot:
            if overlap_with:
                ax = overlap_with
            else:
                fig, ax = plt.subplots(1, figsize = (5,5))
            
            if abs(r[0]-r[1]) < 1e-6 and abs(phi[0]-phi[1]) < 1e-6:
                marker = "o"
            else:
                marker = None
                
            ax.plot(r * np.cos(phi), r * np.sin(phi), marker = marker, mec = color, mfc = color, c = color, ms = 4)
                    
        return t_return, r, phi, beta
    
    def _get_cycle(self, r, phi):
        '''
        Given two arrays of [r] and [phi] for an oscillator in the phase space,
        slice the arrays to get the values over one period of oscillation.
        '''
        
        # TODO: Fails to work properly sometimes. 
        
        if np.abs(r[1]-r[0]) < 1e-3 and np.abs(phi[1]-phi[0]) < 1e-3:
            return r[0], phi[0]
        
        increase = False
        if phi[1]>phi[0]:
            increase = True
        mark = 0
        j = 0
        for i in range(1, len(phi) - 1):
            j += 1
            if (phi[i-1]-phi[i])*(phi[i]-phi[i+1]) < 0:
                mark += 1
            if mark == 2 and increase and phi[i+1]>phi[0]:
                break
            if mark == 2 and not(increase) and phi[i+1]<phi[0]:
                break
        return r[:j+1], phi[:j+1]

    def shortcut(self, rho_0, tau, trajectory_func = None, special = None, err_tol_b = 1e-3, timepoints = 101,
                 maxiter = 100, calculate_metrics = True, report = True, plot_resulting_trajectory = False, save_to_osc = False,
                 t_end_ss_search = 100, t_end_ss_short_search = 100, 
                 **trajectory_func_kwargs):
        
        '''
        ## INTRODUCTION
        
        Shortcut-to-synchronization driving. 
        
        Consider first a the dynamics of the system under a constant driving amplitude specified by constant 
        ``vdp.Omega_1`` and ``vdp.Omega_2``. Under this dynamics a time $T$ is needed for the oscillator system 
        to reach its synchronized, steady state ``rho_ss``. However, it is possible to reach ``rho_ss`` in a less
        time ``tau``<``T`` using time-dependent driving ``Omega_1(t)`` and ``Omega_2(t)``, which are designed with
        the goal *to bring the system to the steady state at ``t``=``tau``*. The shortcut driving will take the 
        system sufficiently close to the steady-state. For ``t``>``tau``, the original driving is applied back to
        the system to drive the system the rest of the way towards the synchronized state.
        
        The algorithm design is inspired by Impens2023 (https://doi.org/10.1038/s41598-022-27130-w), whose choice
        of trajectory is included in this method. 
        
        The algorithm boils down to reverse engineering. As an oscillator is described in terms of the position 
        ``x`` and momentum ``p`` making the phase point, the appropriate quantity is the expectation value of the 
        ladder operators. Here ``<b> = <x> + i<p>`` is used, herein called ``b``. First, ``b`` is set to be the 
        steady-state value ``b_ss`` at ``t``=``tau`` and the initial value ``b_0`` at ``t``=``0``. Then, the
        trajectory is made between these points. The equation of motion for ``b`` is then used to calculate ``Omega_1``
        and ``Omega_2`` for every time point between ``t``=``0`` and ``t``=``tau``. 
        
        The resulting ``Omega_1`` and ``Omega_2`` are then used to evolve the system until its state at 
        time ``t``=``tau``. ``b`` at ``tau`` which is ``b_tau`` is then calculated. ``b_tau`` will be slightly different from
        ``b_ss``. To get a better driving, the offset is subtracted from the current target ``b_target``, which is ``b_ss``
        in the first iteration. The trajectory is then remade with the new target, and the new driving is made with
        respect to this trajectory. The process is repeated until the resulting gets smaller than the specified tolerance 
        or until the specified maximum iteration has been reached, in which case the resulting ``Omega_1`` and ``Omega_2`` 
        are output. 
        
        A more thorough discussion is available in [Research2_Notes].
        
        [CAUTION]
        The Hamiltonian can not be time dependent, as ``qutip.steadystate`` is called in this method.
        
        ---
        
        ## PARAMETERS
        
        ``rho_0``   : 
            The initial density matrix.
            
        ``tau`` :
            Shortcut target time. The resulting driving are designed to terminate here.
        
        ``trajectory_func`` :
            A callable which takes three positional arguments : ``b_0``, ``b_target``, and
            ``timelst``, as well as optional keyword arguments ``**kwargs``. Here ``b_0`` is
            the initial expectation value of the annihilation operator, ``b_target`` is the
            target expectation value of the annihilation operator which equals the steady-state
            value in the first iteration, and ``timelst`` is a time list for the evolution of 
            ``b`` which terminates at ``tau``. Meanwhile, ``**kwargs`` may contain any additional
            arguments passed to the function for variations. ``trajectory_func`` allows the user to
            design his/her own trajectory.
        
        ``special`` :
            There are also premade trajectories which one may use right away using this argument. More
            details in the next section below. If a valid premade function name is input, then the input
            of ``trajectory_func`` will be overriden. The function name is input as a ``str``.
        
        ``err_tol_b``   :
            Error tolerance for ``b``. If the offset is below this value, then the iteration is stopped.
            
        ``timepoints``  : ``101``
            The number of timepoints for the trajectory, and consequently for the driving. The time list 
            passed into the trajectory function is one made using ``numpy.linspace(0, tau, timepoints)``.
            There might be special constraints on this parameter, depending on the trajectory used. For
            example, see ``linear_impens`` in the next section.
            
        ``maxiter`` : ``100``
            The maximum number of iterations. If this number is reached, the method stops, whether the
            offset has fallen below the tolerance or not, and the driving obtained in the final iteration 
            is output.
            
        ``calculate_metrics`` : ``True``
            Calculate the resulting metrics of the shortcut and save it to a ``shortcut_metrics`` class. 
            If this is ``False``, then this method outputs ``None`` instead of the ``shortcut_metrics`` class.
            
        ``report``  : ``True``
            Print out the calculation report in the terminal. If this is specified then ``calculate_metrics``
            is taken to be ``True``. 
            
        ``plot_resulting_trajectory``   : ``False``
            Plot the original target trajectory, the target trajectory corrected for the iteration, and the resulting
            trajectory tracked by ``b`` from the resulting driving in the iteration, for every iteration. Note that
            the method calls ``matplotlib.pyplot.show()`` at every iteration if this argument is ``True``.
            
        ``save_to_osc`` : ``False``
            Save the resulting driving to the ``vdp`` object. If ``False``, then ``vdp.Omega_1`` and ``vdp.Omega_2`` are
            their original values before this method is called. If ``True`` and ``report = True``, then the report will
            also add that the driving are saved into the ``vdp`` object. One will still need to append the default driving
            with the length of the rest of the evolution time list. Since this is up to the user, the method can not to this.
        
        ``t_end_ss_search``   : ``100``
            The end of the time list used to search for the time needed to reach the steady state under constant driving with amplitude equal to
            the average of the total shortcut driving from ``0`` until ``tau``. The time list created has a spacing between two points equal to the
            spacings in the time list used for the trajectory making, i.e. ``np.linspace(0, tau, timepoints)``.
            
        ``t_end_ss_short_search``   :   ``100``
            The end of the time list used to search for the time needed to reach the steady state under the shortcut driving. The time list made
            is concatenated with the trajectory time list generated with ``np.linspace(0, tau, timepoints)``. As such, the final time list
            input to mesolve ends at ``t_end_ss_short_search``+``tau``. The final time list has uniform spacing equal to the spacing in the
            trajectory time list.  
            
        ``**trajectory_func_kwargs`` :
            Optional keyword arguments passed to ``trajectory_func`` or ``special`` as the ``**kwargs`` argument. 
            For ``special``, see the next section.
            
        ---
        
        ## Returns
        
        Returns the resulting shortcut driving as a tuple (Omega_1, Omega_2). Each entry is an array of length ``timepoints``
        whose corresponding time list may be given by ``numpy.linspace(0, tau, timepoints)``.
        
        ---
        
        ## ``Special``: Premade Trajectory Functions
        
        The following are the premade shortcut trajectories available in ``shortcut_trajectories.py`` of this module.
        
        ### ``"linear_impens"`` (https://doi.org/10.1038/s41598-022-27130-w)
        
        The trajectory consists of two straight lines. The first line goes from ``b_0`` to some intermediate point ``b_i``, 
        and the second goes from ``b_i`` to ``b_target``. This intermediate point is the midpoint plus a shift in the 
        imaginary part, i.e. ``bm``=``0.5*(b_0+b_target) + 1j*dy``. 
        
        Optional argument(s): 
        
        ``dy``
            Shift the imaginary part of the intermediate point from the midpoint between ``b_0`` and ``b_target``.  
        
        ### ``"hyperbolic_spiral"``  (Hendry, 2023)
        
        The trajectory is a special hyperbolic spiral defined by the parametric equations (with ``b(t)``=``x(t)+1j*y(t)``)
        
            ``x(t) = r_i * cos(w*t + phi_i) / (at + 1)``
            
            ``y(t) = r_i * cos(w*t + phi_i) / (at + 1)``
            
        where 

            ``w = (1/tau) * (phi_f - phi_i)``
            
            ``a = (1/tau) * (r_i/r_f - 1)``
            
        determines the shape of the spiral. It is also possible for the resulting trajectory 
        to be linear, i.e. when w = 0. 
        
        The direction of the spiral can be swapped by abusing the cyclic property of the angles to change
        the sign of w.
        
        Optional argument(s): 
        
        ``swap_direction``
            Swap the direction of the spiral from whatever original direction is calculated.
        
        
        '''
        
        special_dict = {"linear_impens" : linear_impens,
                        "hyperbolic_spiral" : hyperbolic_spiral}
        
        if special in special_dict:
            trajectory_func = special_dict[special]
        
        original_Omega_1 = copy.deepcopy(self.Omega_1)
        original_Omega_2 = copy.deepcopy(self.Omega_2)
            
        Ham, c_ops = self.dynamics()
        
        b = qt.destroy(self.N)
        b_0 = qt.expect(b, rho_0)
        
        rho_ss = qt.steadystate(Ham, c_ops)
        b_ss = qt.expect(b, rho_ss)
        
        # TODO: Check if ``scale`` argument is needed. It multiplies ``b_0`` and ``b_ss`` so that the resulting 
        #       trajectory is scaled by ``scale``. As it stands, this method does not output the same initial and
        #       final points as that output by ``adler``. This is because the result in ``adler`` is multiplied
        #       by sqrt(2). 
        
        timelst = np.linspace(0, tau, timepoints)
        b_target = b_ss
        iters = 0
        while True:
            iters += 1
            
            b_trajectory, db_trajectory = trajectory_func(b_0, b_target, timelst, **trajectory_func_kwargs)
            
            if iters == 1:
                b_og_trajectory = b_trajectory
            
            Omega = -db_trajectory + (1j*self.Delta + self.gamma_1/2 - self.gamma_2*np.abs(b_trajectory)) * b_trajectory
            
            self.Omega_1 = np.imag(Omega)
            self.Omega_2 = np.real(Omega)
            Ham, c_ops = self.dynamics()
            
            rho_control = self.evolve(rho_0, timelst)
            
            if plot_resulting_trajectory:
                b_control = qt.expect(b, rho_control)
                plt.title(f"iteration {iters}")
                plt.plot(np.real(b_og_trajectory), np.imag(b_og_trajectory), label = "original target trajectory")
                plt.plot(np.real(b_trajectory), np.imag(b_trajectory), label = "corrected target trajectory")
                plt.plot(np.real(b_control),np.imag(b_control), label = "resulting control trajectory")
                plt.scatter(np.real(b_og_trajectory[0]), np.imag(b_og_trajectory[0]), c = "r", label = "start")
                plt.scatter(np.real(b_og_trajectory[-1]), np.imag(b_og_trajectory[-1]), c = "m", label = "finish")
                plt.scatter(np.real([b_trajectory[0], b_control[0]]), np.imag([b_trajectory[0], b_control[0]]), c = "r")
                plt.scatter(np.real([b_trajectory[-1], b_control[-1]]), np.imag([b_trajectory[-1], b_control[-1]]),c = "m")
                plt.legend(loc = "best")
                plt.show()
        
            rho_tau = rho_control[-1]
            b_tau = qt.expect(b, rho_tau) 
            offset_b = b_tau - b_ss
            
            if (iters == maxiter) or (abs(offset_b) < err_tol_b):
                break
            else:
                b_target -= offset_b

        Omega_1_out = self.Omega_1.copy()
        Omega_2_out = self.Omega_2.copy()
        
        report_class = None
            
        if calculate_metrics or report:
        
            Omega_avg = trapz(y = np.abs(Omega_1_out)+np.abs(Omega_2_out), x = timelst) / tau
            amplitude_ratio = Omega_avg / (np.abs(original_Omega_1)+np.abs(original_Omega_2))
            
            d_tr_0_ss = qt.tracedist(rho_0, rho_ss)
            d_tr_tau_ss = qt.tracedist(rho_tau, rho_ss)
            fractional_tracedist_reduction = (d_tr_0_ss - d_tr_tau_ss) / d_tr_0_ss
            
            dt = timelst[1]-timelst[0]
            timelst_ss_search = np.arange(0, t_end_ss_search, dt)
            timelst_ss_short_search = np.arange(tau, t_end_ss_short_search+tau, dt)
            self.Omega_1 = original_Omega_1
            self.Omega_2 = original_Omega_2
            Ham, c_ops = self.dynamics()
            T_ss = qdistance_to_ss(Ham = Ham,
                                c_ops = c_ops,
                                rho0 = rho_0,
                                timelst = timelst_ss_search,
                                dist_func = qt.tracedist,
                                steadystate = rho_ss,
                                _stop_at_t_ss = True)[1]
            l = len(timelst_ss_short_search)
            self.Omega_1 = np.concatenate((Omega_1_out, np.full(shape=(l-1,), fill_value=original_Omega_1)))   
            self.Omega_2 = np.concatenate((Omega_2_out, np.full(shape=(l-1,), fill_value=original_Omega_2)))
            Ham, c_ops = self.dynamics()
            T_ss_short = qdistance_to_ss(Ham = Ham,
                                c_ops = c_ops,
                                rho0 = rho_0,
                                timelst = np.concatenate((timelst, timelst_ss_short_search[1:])), # The first value in the time list is ``tau`` 
                                                                                                # so it is not included in the 
                                                                                                # additional driving.
                                dist_func = qt.tracedist,
                                steadystate = rho_ss,
                                _stop_at_t_ss = True)[1]
            speed_up_ratio = T_ss / T_ss_short
            
            FoM = speed_up_ratio * fractional_tracedist_reduction / amplitude_ratio
            
            report_class = shortcut_metrics(T_ss,
                                            T_ss_short,
                                            amplitude_ratio,
                                            fractional_tracedist_reduction,
                                            speed_up_ratio,
                                            FoM)
            if report:
                s = "== Shortcut finished == \n\n"
                
                if iters < maxiter:
                    s += f"Result obtained with {iters} iterations. \n ===== \n"
                else:
                    s += f"Maximum iterations of {maxiter} is reached. \n ===== \n"
                    
                s += "Specifications: \n"
                
                if special:
                    ss = f"{special} (premade, see pyqosc.{special})"
                else:
                    try:
                        trajectory_func_name = trajectory_func.__name__
                    except:
                        trajectory_func_name = "[unknown]"
                    ss = f"{trajectory_func_name} (input by user)"
                s += f"   Trajectory function: {ss} \n"
                
                s += f"   tau = {tau} \n ===== \n"
                
                s += f"Calculated metrics: \n"
                s += f"   Final (b_tau - b_ss) = {offset_b}\n"
                s += f"   Time to reach the steady state:\n" 
                s += f"      Without shortcut = {T_ss} \n"
                s += f"      With shortcut = {T_ss_short} \n"
                s += f"   Average total amplitude = {Omega_avg}\n\n"
                s += f"   Amplitude ratio = {amplitude_ratio}\n"
                s += f"   Fractional trace distance reduction = {fractional_tracedist_reduction} \n"
                s += f"   Speed up ratio = {speed_up_ratio}\n\n"
                s += f"   Figure of merit (FoM) = {FoM}"
                
                if save_to_osc:
                    s+= "\n ===== \n Omega_1 and Omega_2 are saved into the [vdp] object."
                    
                print(s)
        
        if save_to_osc:
            self.Omega_1 = Omega_1_out
            self.Omega_2 = Omega_2_out
        else:
            self.Omega_1 = original_Omega_1
            self.Omega_2 = original_Omega_2
        
        return Omega_1_out, Omega_2_out, report_class
    
    def shortcut_dCRAB(self, rho_0, tau, timepoints = 101, n = 5, a_guess = [1,1,1,1,1], rand_max = 10, g = 1, rng_seed = None,
                       t_end_ss_search=50, t_end_ss_short_search=50, report = True, save_to_osc=False, plot_optimized_driving = False,
                       minimize_method = "Powell", minimize_options = {'maxiter' : 9999, 'maxfev' : 9999, 'fatol' : 1e-6},
                       control_basis = "Fourier"):
        
        # TODO: The result is very sensitive to the intiial guess and whatnot. This method might be better with another
        #       form of cost function. 
        
        n_a = n // 2
        a_guess = [1] + a_guess

        rng = np.random.default_rng(seed = rng_seed)
        
        original_Omega_2 = copy.deepcopy(self.Omega_2)
        
        timelst = np.linspace(0, tau, timepoints)
        
        Ham, c_ops = self.dynamics()
        rho_ss = qt.steadystate(Ham, c_ops)
        d_tr_0_ss = qt.tracedist(rho_0, rho_ss)
        
        dt = timelst[1]-timelst[0]
        timelst_ss_search = np.arange(0, t_end_ss_search, dt)
        timelst_ss_short_search = np.arange(tau, t_end_ss_short_search+tau, dt)
        T_ss = qdistance_to_ss(Ham = Ham,
                               c_ops = c_ops,
                               rho0 = rho_0,
                               timelst = timelst_ss_search,
                               dist_func = qt.tracedist,
                               steadystate = rho_ss,
                               _stop_at_t_ss = True)[1]
        l = len(timelst_ss_short_search)
        
        class saveclass:
            def __init__(self):
                self.Omega_2 = 1
                self.T_ss_short = 1
                self.Omega_avg = 1
                self.amplitude_ratio = 1
                self.fractional_tracedist_reduction = 1
                self.speed_up_ratio = 1
                self.FoM = 1      
                        
        def Omega_2_call(t, a):
            
            if control_basis == "Fourier":
                w_rand = rng.random(size = n)    
                
                out = 1
                for i in range(n):
                    w_i = (i+1) * (rand_max/n)
                    if i < n_a:
                        out += a[i+1] * np.cos((1+w_rand[i]) * w_i * t)
                    else:
                        out += a[i+1] * np.sin((1+w_rand[i]) * w_i * t)
                out *= g
                out += a[0] * save.Omega_2
                
            if control_basis == "Chebyshev":
                # TODO: work on this
                pass
                            
            return out
        
        def calculate_fom(Omega_2):
            
            Omega_avg = trapz(y = np.abs(Omega_2), x = timelst)/tau
            amplitude_ratio = Omega_avg / np.abs(original_Omega_2)

            self.Omega_2 = Omega_2
            self.dynamics()
            rho_tau = self.evolve(rho_0, timelst)[-1]
            d_tr_tau_ss = qt.tracedist(rho_tau, rho_ss)
            fractional_tracedist_reduction = (d_tr_0_ss - d_tr_tau_ss) / d_tr_0_ss
            
            self.Omega_2 = np.concatenate((Omega_2, np.full(shape=(l-1,), fill_value = original_Omega_2)))
            Ham, c_ops = self.dynamics()
            T_ss_short = qdistance_to_ss(Ham = Ham,
                                         c_ops = c_ops,
                                         rho0 = rho_0,
                                         timelst = np.concatenate((timelst, timelst_ss_short_search[1:])),
                                         dist_func = qt.tracedist,
                                         steadystate = rho_ss,
                                         _stop_at_t_ss = True)[1]
            speed_up_ratio = T_ss/T_ss_short
            
            FoM = speed_up_ratio * fractional_tracedist_reduction / amplitude_ratio
            
            save.T_ss_short = T_ss_short
            save.Omega_avg = Omega_avg
            save.amplitude_ratio = amplitude_ratio
            save.fractional_tracedist_reduction = fractional_tracedist_reduction
            save.speed_up_ratio = speed_up_ratio
            save.FoM = FoM
            
            return abs(1/FoM) # The absolute value is taken here as an effort to avoid getting a negative optimized FoM.
        
        def cost_func(a):
            
            Omega_2 = Omega_2_call(timelst, a)
            save.Omega_2 = Omega_2

            d_tr_tau_ss = calculate_fom(Omega_2)
            
            return d_tr_tau_ss
        
        save = saveclass()
            
        minimize(cost_func, a_guess,
                method = minimize_method,
                options = minimize_options)
        
        if plot_optimized_driving:
            fig, ax = plt.subplots(1, figsize = (5,5))
            ax.plot(timelst, save.Omega_2)
            plt.show()
          
        if report:
            s = "== Shortcut finished == \n\n"
                
            s += "Specifications: \n"
            
            s += f"   Method: dCRAB \n"
            
            s += f"   tau = {tau} \n ===== \n"
            
            s += f"Calculated metrics: \n"
            s += f"   Time to reach the steady state:\n" 
            s += f"      Without shortcut = {T_ss} \n"
            s += f"      With shortcut = {save.T_ss_short} \n"
            s += f"   Average total amplitude = {save.Omega_avg}\n\n"
            s += f"   Amplitude ratio = {save.amplitude_ratio}\n"
            s += f"   Fractional trace distance reduction = {save.fractional_tracedist_reduction} \n"
            s += f"   Speed up ratio = {save.speed_up_ratio}\n\n"
            s += f"   Figure of merit (FoM) = {save.FoM}"
            
            if save_to_osc:
                s+= "\n ===== \n Omega_1 and Omega_2 are saved into the [vdp] object."
                
            print(s)

        if save_to_osc:
            self.Omega_2 = save.Omega_2
        else:
            self.Omega_2 = original_Omega_2
            
        return save.Omega_2
        
    
    # def shortcut_impens(self, rho_0, tau, dy = None, timepoints = 101, err_tol_b = 1e-3, maxiter = 100, report = True,
    #                     plot_trajectory = False, save_to_osc = False):
    
    #     if not(timepoints & 1):
    #         timepoints += 1     # We make sure the number of timepoints is odd so that tau/2 is always a time point.
    #     midpoint = timepoints//2+1
    #     timelst = np.linspace(0, tau, timepoints)
    #     t_half = timelst[:midpoint]

    #     #####
        
    #     original_Omega_1 = self.Omega_1
    #     original_Omega_2 = self.Omega_2
    #     if isinstance(self.Omega_1, (list, np.ndarray)):
    #         original_Omega_1 = original_Omega_1.copy()
    #     if isinstance(self.Omega_2, (list, np.ndarray)):
    #         original_Omega_2 = original_Omega_2.copy()
        
    #     Ham, c_ops = self.dynamics()
        
    #     b = qt.destroy(self.N)
    #     bbx = b.dag()*b*(b+b.dag())/2
    #     bby = b.dag()*b*(b-b.dag())/(2*1j)
        
    #     b_0 = qt.expect(b, rho_0)

    #     rho_ss = qt.steadystate(Ham, c_ops)
    #     b_ss = qt.expect(b, rho_ss)
    #     bbx_ss = qt.expect(bbx, rho_ss)
    #     bby_ss = qt.expect(bby, rho_ss)
        
    #     #####
        
    #     b_target = b_ss
    #     iters = 0
    #     while True:
    #         iters += 1

    #         b_i = 0.5*(b_target+b_0) + 1j*dy
                
    #         db_firsthalf = 2/tau*(b_i-b_0)
    #         b_firsthalf = b_0 + db_firsthalf * t_half
            
    #         db_secondhalf = 2/tau*(b_target-b_i)
    #         b_secondhalf = b_i + db_secondhalf * t_half
            
    #         b_trajectory = np.concatenate((b_firsthalf, b_secondhalf[1:]))
            
    #         if iters == 1:
    #             b_og_trajectory = b_trajectory.copy()
            
    #         Omega = (1j*self.Delta + self.gamma_1/2 - self.gamma_2*np.abs(b_trajectory)) * b_trajectory
    #         Omega[:midpoint] -= db_firsthalf
    #         Omega[midpoint:] -= db_secondhalf
            
    #         self.Omega_1 = np.imag(Omega)
    #         self.Omega_2 = np.real(Omega)
    #         Ham, c_ops = self.dynamics()
            
    #         rho_control = self.evolve(rho_0, timelst)

    #         if plot_trajectory:
    #             b_control = qt.expect(b, rho_control)
    #             plt.title(f"iteration {iters}")
    #             plt.plot(np.real(b_og_trajectory), np.imag(b_og_trajectory), label = "original target trajectory")
    #             plt.plot(np.real(b_trajectory), np.imag(b_trajectory), label = "corrected target trajectory")
    #             plt.plot(np.real(b_control),np.imag(b_control), label = "resulting control trajectory")
    #             plt.legend(loc = "best")
    #             plt.show()

    #         rho_tau = rho_control[-1]
    #         b_tau = qt.expect(b, rho_tau) 
    #         Delta_b = np.abs(b_tau-b_ss)
            
    #         if (iters == maxiter) or (Delta_b < err_tol_b):
    #             bbx_tau = qt.expect(bbx, rho_tau)
    #             bby_tau = qt.expect(bby, rho_tau)
    #             Delta_bbb = np.sqrt((bbx_tau-bbx_ss)**2+(bby_tau-bby_ss)**2)
    #             break
    #         else:
    #             offset_b = b_tau - b_ss
    #             b_target -= offset_b
        
    #     Omega_1_out = self.Omega_1.copy()
    #     Omega_2_out = self.Omega_2.copy()
    #     if not(save_to_osc):
    #         self.Omega_1 = original_Omega_1
    #         self.Omega_2 = original_Omega_2
            
    #     if report:
    #         s = "== Shortcut finished == \n\n"
    #         if iters < maxiter:
    #             s += f"Result obtained with {iters} iterations. \n\n"
    #         else:
    #             s += "Maximum iterations reached. \n\n"
    #         s += "Specifications: \n"
    #         s += "   Method: Impens2023 \n"
    #         s += f"   tau = {tau} \n"
    #         s += f"   Delta_y = {dy} \n\n"
    #         s += f"Calculated metrics: \n"
    #         s += f"   Delta <b> = {Delta_b} \n"
    #         s += f"   Delta <b*bb> = {Delta_bbb} \n\n"
    #         if save_to_osc:
    #             s+= "Omega_1 and Omega_2 are saved into the [vdp] object."
    #         print(s)
        
    #     return Omega_1_out, Omega_2_out