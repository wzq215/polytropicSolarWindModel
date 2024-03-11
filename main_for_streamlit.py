# Evolution Model of Solar Wind Velocity in PolyTropic System

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def solve_s_c(C0_Cg, gamma):
    '''
    Given C_g/C_0 and gamma, solve for s_c using the following equation:
    (1/2)*((C_g/C_0)**((4)/(gamma-1))*s_c**((2*(2*gamma-3))/(gamma-1))-1)+(1)/(gamma-1)*((C_0**2)/(C_g**2)*s_c-1)-2*(s_c-1)=0
    When (C_g/C_0)**2 < 2(gamma-1), the equation has 1 solution;
    When (C_g/C_0)**2 > 2(gamma-1) and (C_g/C_0)**2 < 1, the equation has 2 solutions;
    When (C_g/C_0)**2 > 2(gamma-1) and (C_g/C_0)**2 > 1, the equation has no solution
    '''
    func = lambda s_c: (1 / 2) * (
                C0_Cg ** (-(4) / (gamma - 1)) * s_c ** ((2 * (2 * gamma - 3)) / (gamma - 1)) - 1) + (1) / (
                                   gamma - 1) * (s_c*(C0_Cg)**2 - 1) - 2 * (s_c - 1)
    # solve the equation
    if (C0_Cg) ** 2 < 2 * (gamma - 1):
        print("one solution at the sonic point")
        s_c = fsolve(func, 1)
        statu = np.isclose(func(s_c), 0.)
    elif (C0_Cg) ** 2 > 2 * (gamma - 1) and (C0_Cg) ** 2 < 1:
        print("two solutions at the sonic point")
        s_c = fsolve(func, [1,10])
        if s_c[1]-s_c[0]<1e-5*s_c[1]:
            s_c = fsolve(func, [1,100])
        s_c.sort()
        statu = np.isclose(func(s_c),[0.,0.])
    else:
        print("no solution at the sonic point")
        s_c = None
        statu=None
    return s_c,statu

def dx(x,t,args):
    '''
    x=V
    :param x: variables to be solved
    :param r: solar centric distance
    :return: derivative of x with respect to r
    '''
    V0, Cg, gamma = args
    r=x[0]
    V=x[1]
    Cs2 = (V0/r**2/V)**(gamma-1)
    dx1 = 1.
    dx2 = (2*Cs2/r-2*Cg**2/r**2)/(V-Cs2/V)
    dxdr = np.array([dx1,dx2])
    return dxdr

# Fourth-order Runge-Kutta method
def rk4(x,t,dt,derivsRK,args=None):
    '''
    :param x: variables to solve
    :param t: time
    :param dt: time step
    :param derivsRK: derivative of x with respect to t
    :return: derivative of x with respect to t
    '''
    # RK4 algorithm
    half_dt = 0.5*dt
    F1 = derivsRK(x,t,args)
    t_half = t + half_dt
    xtemp = x + half_dt*F1
    F2 = derivsRK(xtemp,t_half,args)
    xtemp = x + half_dt*F2
    F3 = derivsRK(xtemp,t_half,args)
    t_full = t + dt
    xtemp = x + dt*F3
    F4 = derivsRK(xtemp,t_full,args)
    xout = x + dt/6.*(F1 + F4 + 2.*(F2+F3))
    return xout

# Given the time step, solve the change over a period of time
def rkdumb(x,t,derivsRK,args=None):
    '''
    :param x: variable to solve
    :param t: time
    :param nsteps: time step
    :param derivsRK: derivative of x with respect to t
    :return: derivative of x with respect to t
    '''
    # Integrate x(t) using RK4 method
    dt = t[1]-t[0]
    nsteps = len(t)-1
    xs = np.zeros((nsteps+1,2))
    xs[0,:] = x
    for i in range(0,nsteps):
        xs[i+1,:] = rk4(xs[i,:],t[i],dt,derivsRK,args)
    return xs

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test solve_s_c
    G = 6.67e-11
    Msun = 1.989e30 #Kg
    Rsun = 695500e3 #m
    kB = 1.380649e-23
    mp = 1.672621637e-27 #kg
    Cg = np.sqrt(G*Msun/2/Rsun)
    s_c_i = 1  # 0 or 1, 0 for deceleration, 1 for acceleration

    gamma = 1.05
    T0 = 2e6

    # Add input fields for key parameters in the left sidebar
    gamma = st.sidebar.slider('gamma', 1.0, 2.0, 1.05)
    T0 = st.sidebar.slider('T0', 1e6, 3e6, 2e6)

    C0 = np.sqrt(gamma*kB*T0/mp)
    # C0_Cg = 0.8
    C0_Cg = C0/Cg
    s_c,statu = solve_s_c(C0_Cg, gamma)
    # show the result
    # print(s_c)

    # All velocities are normalized by C0, all distances are normalized by r0

    V02 = C0_Cg**(-2-4/(gamma-1))*s_c[s_c_i]**((3*gamma-5)/(gamma-1)) # actually V0/C0
    Vc2 = C0_Cg**(-2)/s_c[s_c_i];
    args = [np.sqrt(V02), 1/C0_Cg, gamma]
    x0 = [1., np.sqrt(V02)]
    ts = np.linspace(0., s_c[s_c_i]-1., 801)
    sol1 = rkdumb(x0, ts, dx, args=args)
    eps = 1e-3
    x0 = eps * dx(sol1[-2, :], 0, args) + [s_c[s_c_i], np.sqrt(Vc2)]
    ts = np.linspace(x0[0]-1, 100, 4001)
    sol2 = rkdumb(x0, ts, dx, args=args)
    sol = np.append(sol1, sol2, axis=0)
    # sol1 = odeint(dx, x0, ts, args=(np.sqrt(V02), 1/C0_Cg))
    plt.plot(sol[:, 0], sol[:, 1], 'g')
    plt.xlabel('$r/r_0$', fontsize=13)
    plt.ylabel('$V/C_0$', fontsize=13)
    plt.title('$\gamma=$'+str(gamma)+' $C_0/C_g=$'+"%.2f" % C0_Cg)
    # plt.grid()
    plt.box(on=True)
    st.pyplot()
    plt.show()
    # plt.savefig("test.png")
    print(V02)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
