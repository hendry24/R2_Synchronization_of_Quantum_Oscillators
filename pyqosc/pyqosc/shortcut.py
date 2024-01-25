import numpy as np
import matplotlib.pyplot as plt


class shortcut_metrics:
    def __init__(self, T_ss, T_ss_short, amplitude_ratio, 
                 fractional_tracedist_reduction, speed_up_ratio,
                 FoM):
        self.T_ss = T_ss
        self.T_ss_short = T_ss_short
        self.amplitude_ratio = amplitude_ratio
        self.fractional_tracedist_reduction = fractional_tracedist_reduction
        self.speed_up_ratio = speed_up_ratio
        self.FoM = FoM
        
    def __repr__(self):
        s = "== Calculated Shortcut Metrics =="
        s += f"T_ss = {self.T_ss}"
        s += f"T_ss_short = {self.T_ss_short}"
        s += f"amplitude_ratio = {self.amplitude_ratio}"
        s += f"fractional_tracedist_reduction = {self.fractional_tracedist_reduction}"
        s += f"speed_up_ratio = {self.speed_up_ratio}"
        s += f"FoM = {self.FoM}"
        
        return s
    
    def __str__(self):
        return self.__repr__()

def linear_impens(b_0, b_target, timelst, **kwargs):
    dy = kwargs.get("dy", 0)

    b_i = 0.5*(b_target+b_0) + 1j*dy
    
    l = len(timelst)     # must be odd so that tau/2 is a time point.
    midpoint = l//2+1
    tau = timelst[-1]
    t_half = timelst[:midpoint]
    
    db_firsthalf = np.full(shape=(midpoint,), fill_value=2/timelst[-1]*(b_i-b_0))
    b_firsthalf = b_0 + db_firsthalf * t_half
    
    db_secondhalf = np.full(shape=(midpoint,), fill_value=2/tau*(b_target-b_i))
    b_secondhalf = b_i + db_secondhalf * t_half
    
    b_trajectory = np.concatenate((b_firsthalf, b_secondhalf[1:]))
    db_trajectory = np.concatenate((db_firsthalf, db_secondhalf[1:]))
    return b_trajectory, db_trajectory

def _test_hyperbolic_spiral():
    r_i = 1
    phi_i = np.pi/4

    w = 0.01
    p = 0.2

    t = np.linspace(0, 10, 1001)

    x = r_i * np.cos(w * t + phi_i) / (p *t+1)
    y = r_i * np.sin(w * t + phi_i) / (p *t+1)
    x1 = r_i * np.cos(w * t)
    y1 = r_i * np.sin(w * t)

    fig, ax = plt.subplots(1, figsize = (6,6))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, c = "k", alpha = 0.4)
    ax.axvline(0, c = "k", alpha = 0.4)
    ax.plot(x1, y1, c="k", alpha = 0.4)
    ax.scatter(x[0], y[0])
    ax.scatter(x[-1], y[-1])
    ax.plot(x,y)
    plt.show()
    
def hyperbolic_spiral(b_0, b_target, timelst, **kwargs):
    
    '''
    The equation for a hyperbolic spiral which is useful for this function 
    can generally be written as the parametric equations
    
        x(t) = r_i * cos(w * t + phi_i) / (a * t + 1)
    
        y(t) = r_i * sin(w * t + phi_i) / (a * t + 1)
        
    Note that the spiral starts out (t=0) at the initial polar points 
    (r_i, phi_i). As t increases, the spiral goes inwards. So, the initial
    polar points must be chosen to be one of b_0 and b_target which is
    further out. This can be easily done by comparing their moduli.
    
    Meanwhile, w is a parameter we can interpret as the angular frequency.
    Lastly, we have another parameter a. Together, w and a parameterize 
    the form of the spiral. The larger w is, the more rotations we have for
    a given radius. the larger a is, the faster the radius falls for a given t.
    
    You can play with these parameters and the general equations of the spiral
    by using ``_test_hyperbolic_spiral`` function defined in this file.
    
    To make sure that (x(tau), y(tau)) = (x_f, y_f), the final position, we need to 
    find the appropriate w and a which satisfies this condition. It is easy
    to show that
    
        w = (1/tau) * (phi_f - phi_i)
    
        a = (1/tau) * (r_i/r_f - 1)
        
    where (r_f,phi_f) is one of b_0 and b_target which does not make (r_i,phi_i). 
    
    w may be positive or negative depending on the relation between the initial and final
    phase. Since the sign of w determines the direction of the spiral, it is useful to take
    a deeper look at this parameter. 
    
    If phi_f > phi_i, then w is always positive. 
    If phi_f = phi_i, then w = 0. 
    If phi_f < phi_i, then w is always negative. 
    
    Our goal here is to make a trajectory from the initial to the final point. But
    we have the freedom to change the sign of the angles by adding/subtracting by 2*pi since
    it would result in the same angles. By doing this, we can control the sign of omega and
    hence the direction of the spiral. This is the reasoning behind the ``swap_direction``
    optional argument present in this trajectory making algorithm. By the relations above, we
    can just shift the the value of, say, phi_f until the inequality which flips the sign
    of w is met.
    
    It is also evident that it is impossible to get to the origin within a finite amount of
    time. This is one weakness of this trajectory.
    '''

    r_0 = np.abs(b_0)
    phi_0 = np.angle(b_0)
    
    r_target = np.abs(b_target)
    phi_target = np.angle(b_target)
    
    if r_target>r_0:
        r_i = r_target
        phi_i = phi_target
        
        r_f = r_0
        phi_f = phi_0
    else:
        r_i = r_0
        phi_i = phi_0
        
        r_f = r_target
        phi_f = phi_target
    
    tau = timelst[-1]
    
    d_phi = phi_f - phi_i
    if kwargs.get("swap_direction", False):
        if d_phi > 0:
            while phi_f > phi_i:
                phi_f -= 2*np.pi
        if d_phi < 0:
            while phi_f < phi_i:
                phi_f += 2*np.pi
                
    w = (1/tau) * (phi_f - phi_i)
    a = (1/tau) * (r_i/r_f - 1)
    
    x_trajectory = r_i * np.cos(w * timelst + phi_i) / (a * timelst + 1)
    y_trajectory = r_i * np.sin(w * timelst + phi_i) / (a * timelst + 1)
    
    if r_target>r_0:    # The trajectory must start at b_0. 
        x_trajectory = x_trajectory[::-1]
        y_trajectory = y_trajectory[::-1]
    
    dx_trajectory = np.gradient(x_trajectory, timelst)
    dy_trajectory = np.gradient(y_trajectory, timelst)
    
    b_trajectory = x_trajectory + 1j * y_trajectory
    db_trajectory = dx_trajectory + 1j * dy_trajectory
    
    return b_trajectory, db_trajectory

