import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import scipy as sp

options = qt.Options(nsteps = int(1e6))
plt.rcParams.update({"font.size" : 13})
    
################################################################################################################################################################
################################################################################################################################################################

def steady_state(Ham, c_ops, plot_wigner = False, xlim = 6, ylim = 6, overlap_with = None):
    
    rho_ss = qt.steadystate(Ham, c_ops)
    
    if plot_wigner:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (5, 5))
        x = np.linspace(-xlim, xlim, 31)
        y = np.linspace(-ylim, ylim, 31)
        plot = ax.contourf(x, y, qt.wigner(rho_ss, x, y), 100, cmap = "viridis")
        fig.colorbar(plot, ax = ax)
        ax.set_aspect("equal")
    
    return rho_ss

################################################################################################################################################################
################################################################################################################################################################

def qdistance_to_ss(Ham, c_ops, rho_0, timelst, dist_func = qt.fidelity, steadystate = None, 
                    plot = False, overlap_with=None, ss_err_tol = 1e-3,_stop_at_t_ss = False):

    if plot:
        if not(overlap_with):
            fig, ax = plt.subplots(1, figsize = (5, 4))
        else:
            ax = overlap_with
            
    if not(steadystate):
        rho_ss = steady_state(Ham, c_ops)
    else:
        rho_ss = steadystate
        
    rho_t = qt.mesolve(Ham, rho_0, timelst, c_ops, options = options).states
    
    dist_lst = []
    mark = True
    for i in range(len(timelst)):
        dist_lst.append(dist := dist_func(rho_t[i], rho_ss))
        
        # Mark when steady state is reached.
        if mark and abs(dist - dist_func(rho_ss, rho_ss)) < ss_err_tol:
            mark = False
            t_ss = timelst[i]
            if _stop_at_t_ss:
                break
            if plot:
                ax.axvline(timelst[i], ls = ":", c = 'r', label = f"steady state reached at \n t = {round(timelst[i], 2)}")
    
    if plot:
        
        name_dict = {qt.fidelity : "State fidelity",
                     qt.bures_angle : "Bures angle",
                     qt.bures_dist : "Bures distance",
                     qt.hellinger_dist : "Hellinger distance",
                     qt.hilbert_dist : "Hilbert-Schmidt distance",
                     qt.tracedist : "Trace distance"}
        
        if not(overlap_with):
            ax.set_xlabel("time")
            ax.set_ylabel(f"{name_dict.get(dist_func, 'Distance')} wrt. steady-state")
            ax.set_xlim(0, timelst[-1])
            ax.set_ylim(0, 1.05)
                        
        ax.plot(timelst, dist_lst, c = 'b')
        ax.axhline(1.0, c = "k", ls = ":", alpha = 0.5)
        ax.legend(loc = "best")
        
    return dist_lst, t_ss

################################################################################################################################################################
################################################################################################################################################################

def ss_c_phasedist(rho_ss, late_r_cycle, late_phi_cycle, num_bins = 36, plot = False, overlap_with = None):
    '''
    Plot the probability histogram corresponding ot the expectation value of the oscillator
    given by the late-time dynamics (``late_r_cycle``, ``late_phi_cycle``) in the phase space. 
    The probability is taken with respect to the steady-state Wigner function. 
    
    If there are mre than one points over a given interval of phi, then the positions are 
    averaged, then the Wigner function is evaluated with respect to that point.
    
    ----------
    Returns
    ----------
    Positions of histogram midpoints and the corresponding values, ``phi_bin_midpoints, hist_data``.
    
    ----------
    Parameters
    ----------
    
    ``rho_ss``  :
        The steady-state density matrix.
        
    ``late_r_cycle``  :
        Late-time values for r over one cycle. Get from e.g. ``vdp.adler(...,one_cycle=True)``.
        
    ``late_phi_cycle``    :
        Late-time values for phi over one cycle. Get from e.g. ``vdp.adler(...,one_cycle=True)``.
        
    ``num_bins``    :   36
        Number of histogram bins.
        
    ``overlap_with``    : None
        ``matplotlib.axes.Axes``object to plot in. By default, create a new figure and axis to
        plot in.
        
    '''
    
    phi_hist = np.linspace(0, 2*np.pi, num_bins+1)
    phi_bin_midpoints = np.linspace(phi_hist[1]/2, 2*np.pi-phi_hist[1]/2, num_bins)
    
    hist_data = []
    
    ignore_lst = []
    for i in range(num_bins):
        x = y = 0
        get_index = []
        for j in range(len(late_phi_cycle)):
            if j in ignore_lst:
                continue
            
            if phi_hist[i] < late_phi_cycle[j] < phi_hist[i+1]:            
                get_index.append(j)
                ignore_lst.append(j)
        
        if get_index:
            for index in get_index:
                x += late_r_cycle[index] * np.cos(late_phi_cycle[index])
                y += late_r_cycle[index] * np.sin(late_phi_cycle[index])
            x /= len(get_index)
            y /= len(get_index)
        
            hist_data.append(qt.wigner(rho_ss, x, y)[0][0])
        else:
            hist_data.append(0.0)
    
    # normalize
    hist_data = np.array(hist_data)
    hist_data /= np.sum(hist_data)
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (10, 5))
        
        ax.bar(phi_bin_midpoints, hist_data, width = phi_bin_midpoints[1]-phi_bin_midpoints[0], ec = "k", color = "b", label = "cl")
        
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    
    return phi_bin_midpoints, hist_data

################################################################################################################################################################
################################################################################################################################################################

def ss_q_phasedist(rho_ss, num_bins, plot = False, overlap_with = None):
    N = rho_ss.dims[0][0]

    bin_width = 2*np.pi / num_bins
    phi_hist_midpoints = np.linspace(bin_width/2, 2*np.pi - bin_width/2, num_bins)
    
    hist_data = []
    for i in range(num_bins):
        phi = phi_hist_midpoints[i]
        phi_ket = 0
        for n in range(N):
            phi_ket += np.exp(1j * n * phi) * qt.basis(N, n)
        hist_data.append(qt.expect(rho_ss, phi_ket))    # No need to divide by 2pi as the array will be normalized below.
    
    hist_data = np.array(hist_data)
    hist_data /= np.sum(hist_data)
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (10, 5))
        
        ax.bar(phi_hist_midpoints, hist_data, width = bin_width, ec = "k", color = "r", label = "qm")
        
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    
    return phi_hist_midpoints, hist_data

################################################################################################################################################################
################################################################################################################################################################

def ss_q_spectrum(Ham, c_ops, b, omega = np.linspace(-1, 1, 101), 
               plot = False, overlap_with = None, label = r"qm"):
    
    spect = qt.spectrum(Ham, omega, c_ops, b, b.dag())
    spect /= np.max(spect)
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (5, 4)) 
        
        ax.plot(omega, spect, label = label, ls = "--")
        ax.legend(loc = "best")
        ax.set_ylabel(r"$S(\omega)$")
    
    return omega, spect, omega[spect == np.max(spect)][0]

################################################################################################################################################################
################################################################################################################################################################

def ss_c_acf(timelst, expval, plot = False, overlap_with = None, **plot_kwargs):
    
    l = len(timelst)
    tlag = [timelst[0] - timelst[-i] for i in range(1, l+1)] + [timelst[i]-timelst[0] for i in range(1, l)]
    
    acf = sp.signal.correlate(expval, expval, mode = "full")
    # The function returns the autocorrelation of beta with respect to lag. The center of
    # the list corresponds to zero lag, while the ends correspond to maximum lag with which
    # the autocorrelation is not zero. The mode used is "full"
    # since we need the values over all lags to get the best spectral density function.
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (5,4))
                
        ax.plot(tlag, np.real(acf), label = r"Re{acf}")
        ax.plot(tlag, np.imag(acf), label = r"Im{acf}")
        ax.set_xlabel(r"lag $\tau$")
        
        ax.legend(loc = "best")
            
    return tlag, acf
    
def ss_c_spectrum(timelst_ss, expval_ss, omega_lim = 1.0, plot = False, plot_bar = False, 
                  overlap_with = None, **plot_kwargs):
    
    tlag, acf = ss_c_acf(timelst_ss, expval_ss)
    
    n = len(tlag)
    T = (tlag[-1] - tlag[0]) / n

    omega = sp.fft.fftfreq(n, T) * 2 * np.pi

    spect = np.abs(sp.fft.fft(acf))
    spect /= np.max(spect)
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (5, 4))
        
        if plot_bar:
            ax.bar(omega, spect, **plot_kwargs)
        else:
            ax.plot(omega, spect, **plot_kwargs)
            
        ax.legend(loc = "best")
        ax.set_ylabel(r"$S(\omega)$")
        ax.set_xlim(-omega_lim, omega_lim)

    return omega, spect, omega[np.where(spect==np.max(spect))][0]

################################################################################################################################################################
################################################################################################################################################################

def ss_qsl_mt(Ham, c_ops, rho_0, timelst, plot = False, overlap_with = None):
    
    rho_ss = qt.steadystate(Ham, c_ops)
    
    rho_t = qt.mesolve(Ham, rho_0, timelst, c_ops).states
    
    l = len(timelst)
    dt = timelst[1]-timelst[0]
    mean_stdev_t = [np.nan]
    s = 0
    stdev0 = np.sqrt(qt.variance(Ham, rho_t[0]))
    for i in range(1,l):
        stdev = np.sqrt(qt.variance(Ham, rho_t[i]))
        mean_stdev_t.append((dt/(timelst[i]-timelst[0])) * (stdev0/2 + s + stdev/2))    # Usual trapz formula, divided by the total time
        s += stdev
    
    bures_angle_t = [qt.bures_angle(rho_t[i], rho_ss) for i in range(l)]
        
    qsl_t = [0]
    for i in range(1,l):
        qsl_t.append(bures_angle_t[i] / mean_stdev_t[i])
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (5, 4), constrained_layout = True)
            ax.set_ylabel(r"$\tau_\mathrm{QSL}$")
            ax.set_xlabel(r"$t$")
        ax.plot(timelst[1:], qsl_t[1:])
        
    return qsl_t

################################################################################################################################################################
################################################################################################################################################################

# def ss_qsl_funo(qosc, rho_0, init_tau = 1, fsolve_xtol = 1e-3, 
#                 fsolve_maxfev = int(1e6), mesolve_timepoints = 101, quad_limit = 101):
    
#     N = qosc.N
#     Ham, c_ops = qosc.dynamics()
#     rho_ss = qt.steadystate(Ham, c_ops)
    
#     d_tr = qt.tracedist(rho_0, rho_ss)
    
#     def rho(t):
#         '''Get density matrix at time t using mesolve'''
#         return qt.mesolve(Ham, rho_0, np.linspace(0, t, mesolve_timepoints), c_ops, options = qt.Options(nsteps=int(1e9))).states[-1]
                    
#     def D(t):
#         s = 0
#         rho_t = rho(t)
#         for c_op in c_ops:
#             s += c_op * rho_t * c_op.dag() - 0.5 * qt.commutator(c_op.dag()*c_op, rho_t, kind = "anti")
#         return s, rho_t
    
#     def H_D(t):
#         D_t, rho_t = D(t)
#         rho_eigvals, rho_eigstates = rho_t.eigenstates()
#         out = 0
#         for m in range(N):
#             pm = rho_eigvals[m]
#             bm = rho_eigstates[m]
#             for n in range(N):
#                 pn = rho_eigvals[n]
#                 if pn==pm:
#                     continue
#                 bn = rho_eigstates[n]
#                 out += D_t.matrix_element(bm, bn) / (pn-pm) * bm * bn.dag()
#         out *= 1j
#         return out, rho_t
        
#     def stdev_sum(t):
#         H_D_t, rho_t = H_D(t)
#         return np.sqrt(qt.variance(Ham, rho_t)) + np.sqrt(qt.expect(H_D_t**2, rho_t))
    
#     #####
    
#     def W(t):
#         rho_t = rho(t)
#         rho_eigvals, rho_eigstates = rho_t.eigenstates()
        
#         Wmn = np.empty(shape=(len(c_ops),N,N))
#         # Assuming [gamma] is independent of [omega], [W_{mn}^{omega,alpha}] and [W_{nm}^{-omega,alpha}]
#         # are identical.
        
#         for i in range(len(c_ops)):
            
#             omega_is_0 = False
#             if qt.commutator(c_ops[i],Ham)==0:
#                 omega_is_0 = True
            
#             for m in range(N):
#                 bm = rho_eigstates[m]
                
#                 for n in range(N):
#                     if m==n and omega_is_0:
#                         Wmn[i][m][n] = 0
                    
#                     bn = rho_eigstates[n]
                    
#                     Wmn[i][m][n] = np.abs(bm.overlap(c_ops[i]*bn))**2
                    
#         return rho_eigvals, Wmn
    
#     def sigma_and_A(t):
#         p, Wmn = W(t)
#         sigma = 0
#         A = 0
#         for i in range(len(c_ops)):
#             for m in range(N):
#                 for n in range(N):
#                     if p[m]*p[n]>0:
#                         sigma += Wmn[i][m][n] * p[n] * np.log(p[n]/p[m])
#                     A += (p[n]+p[m]) * Wmn[i][m][n]
#         return sigma, A
                
#         #####
#     def funo(tau):
        
#         def time_quad(func):
#             return sp.integrate.quad(func, 0, tau, limit = quad_limit)[0]
        
#         timelst = np.linspace(0, tau, quad_limit).flatten()
#         sigma_lst = np.empty(shape=(quad_limit,))
#         A_lst = np.empty(shape=(quad_limit,))
#         sigma_lst[0] = 0
#         A_lst[0] = 0
#         for i in range(1, quad_limit):
#             sigma_lst[i], A_lst[i] = sigma_and_A(float(timelst[i])) # Need to convert to float or qutip mesolve won't work.
        
#         def time_trapz(x, y):
#             return sp.integrate.trapz(y=y, x=x)
         
#         qsl = time_quad(stdev_sum) + np.sqrt(0.5 * time_trapz(timelst, sigma_lst) * time_trapz(timelst, A_lst)) - d_tr
        
#         if qsl <= 0:
#             raise ValueError("Invalid value of QSL is obtained. Algorithm fails.")
        
#         return qsl 
        
#     return sp.optimize.fsolve(funo, init_tau, xtol = fsolve_xtol, maxfev = fsolve_maxfev)[0]

def ss_qsl_funo(qosc, rho_0, init_tau = 1, fsolve_xtol = 1e-3, 
                fsolve_maxfev = int(1e6), mesolve_timepoints = 101, quad_limit = 101):
    
    N = qosc.N
    Ham, c_ops = qosc.dynamics()
    rho_ss = qt.steadystate(Ham, c_ops)
    
    d_tr = qt.tracedist(rho_0, rho_ss)
    
    def rho(t):
        '''Get density matrix at time t using mesolve'''
        return qt.mesolve(Ham, rho_0, np.linspace(0, t, mesolve_timepoints), c_ops, options = qt.Options(nsteps=int(1e9))).states[-1]
                    
    def D(t):
        s = 0
        rho_t = rho(t)
        for c_op in c_ops:
            s += c_op * rho_t * c_op.dag() - 0.5 * qt.commutator(c_op.dag()*c_op, rho_t, kind = "anti")
        return s, rho_t
    
    def tr_H_D_2_rho(t):
        D_t, rho_t = D(t)
        rho_eigvals, rho_eigstates = rho_t.eigenstates()
        out = 0
        for m in range(N):
            pm = rho_eigvals[m]
            bm = rho_eigstates[m]
            for n in range(N):
                pn = rho_eigvals[n]
                if pn==pm:
                    continue
                bn = rho_eigstates[n]
                out += pm * D_t.matrix_element(bm, bn) * D_t.matrix_element(bn, bm) / (pm-pn)**2
        return out, rho_t
        
    def stdev_sum(t):
        tr_H_D_2_rho_t, rho_t = tr_H_D_2_rho(t)
        return np.sqrt(qt.variance(Ham, rho_t)) + np.sqrt(tr_H_D_2_rho_t)
    
    #####
    
    def W(t):
        rho_t = rho(t)
        rho_eigvals, rho_eigstates = rho_t.eigenstates()
        
        Wmn = np.empty(shape=(len(c_ops),N,N))
        # Assuming [gamma] is independent of [omega], [W_{mn}^{omega,alpha}] and [W_{nm}^{-omega,alpha}]
        # are identical.
        
        for i in range(len(c_ops)):
            
            omega_is_0 = False
            if qt.commutator(c_ops[i],Ham)==0:
                omega_is_0 = True
            
            for m in range(N):
                bm = rho_eigstates[m]
                
                for n in range(N):
                    if m==n and omega_is_0:
                        Wmn[i][m][n] = 0
                    
                    bn = rho_eigstates[n]
                    
                    Wmn[i][m][n] = np.abs(bm.overlap(c_ops[i]*bn))**2
                    
        return rho_eigvals, Wmn
    
    def sigma_and_A(t):
        p, Wmn = W(t)
        sigma = 0
        A = 0
        for i in range(len(c_ops)):
            for m in range(N):
                for n in range(N):
                    if p[m]*p[n]>0:
                        sigma += Wmn[i][m][n] * p[n] * np.log(p[n]/p[m])
                    A += (p[n]+p[m]) * Wmn[i][m][n]
        return sigma, A
                
        #####
    def funo(tau):
        
        def time_quad(func):
            return sp.integrate.quad(func, 0, tau, limit = quad_limit)[0]
        
        timelst = np.linspace(0, tau, quad_limit).flatten() # fsolve somehow makes this a nested array, so flattening is needed.
        sigma_lst = np.empty(shape=(quad_limit,))
        A_lst = np.empty(shape=(quad_limit,))
        sigma_lst[0] = 0
        A_lst[0] = 0
        for i in range(1, quad_limit):
            sigma_lst[i], A_lst[i] = sigma_and_A(float(timelst[i])) # Need to convert to float or qutip mesolve won't work.
        
        def time_trapz(x, y):
            return sp.integrate.trapz(y=y, x=x)
         
        qsl = time_quad(stdev_sum) + np.sqrt(0.5 * time_trapz(timelst, sigma_lst) * time_trapz(timelst, A_lst)) - d_tr
        
        if qsl <= 0:
            raise ValueError("Invalid value of QSL is obtained. Algorithm fails.")
        
        return qsl
        
    return sp.optimize.fsolve(funo, init_tau, xtol = fsolve_xtol, maxfev = fsolve_maxfev)[0]

################################################################################################################################################################
################################################################################################################################################################

def ss_qsl_delcampo(qosc, rho_0, init_tau = 1, fsolve_xtol = 1e-3, 
                    fsolve_maxfev = int(1e6), mesolve_timepoints = 101, quad_limit = 101):
    
    # TODO: Incorporate time-dependent Linbladian.
    
    Ham, c_ops = qosc.dynamics()
    rho_ss = qt.steadystate(Ham, c_ops)
    
    Ldag_rho_0 = 1j * qt.commutator(Ham, rho_0)
    for i in range(len(c_ops)):
        F = c_ops[i]
        Ldag_rho_0 += F.dag()*rho_0*F - 0.5 * qt.commutator(F.dag()*F, rho_0, "anti")
        
    fig_merit = ((rho_0*rho_ss).tr())/((rho_0**2).tr())
    tau_qsl = np.abs(fig_merit-1)*(rho_0**2).tr()/np.sqrt((Ldag_rho_0**2).tr())
    
    return tau_qsl
    