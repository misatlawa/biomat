import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt

from data_utils import start_params, experimental_data, fited_values


class Parameterize_ODE():
  # SFd parameters: not fitted to data
  gamma = 14.173  # 0 gamma
  mu = 2.448  # 1 mu
  rho = 0.232,  # 2 rho

  alpha = 2.8e-8,  # 3 alpha
  beta = 0.0132,  # 4 beta
  repair_rate = 2.0358  # 5 repair rate

  def __init__(self):
    self.V_data = experimental_data
    self.E_data = None
    self.x = np.array(range(12,33))
    self.x_data = np.array([12, 17, 22, 27, 32])
    self.experimental_data = experimental_data

    ## INDEXES in p, not  VALUES
    # CCs populations parameters
    self.r = 0  # Viable cancer cells volume doubling time
    self.k = 1  # Tumor carrying capacity
    self.a = 2  # CTLs' killing rate
    self.d = 3  # Clearance rate of dying cells

    self.l = 4  # Decay rate of effector cells
    self.omega = 5  # Baseline T cell recruitment rate
    self.omega2 = 6  # Fold change in the baseline T cell recruitment rate due to immunogenic cell death
    self.e = 7  # Initial fold change in recruitment of cytotoxic T cells caused by immunotherapy
    self.clr = 8  # 9H10 immunotherapy clearance rate
    self.E = 9
    self.D_prop = 10

    #MCA38
    # CCs populations parameters
    self.r_MCA = 11  # Viable cancer cells volume doubling time
    self.k_MCA = 12  # Tumor carrying capacity
    self.a_MCA = 13  # CTLs' killing rate
    self.d_MCA = 14  # Clearance rate of dying cells

    self.omega_MCA = 15  # Baseline T cell recruitment rate
    self.E_MCA = 16
    self.D_prop_MCA = 17



  @staticmethod
  def G(x):
    return 2 * (x + np.exp(-x) - 1) / (x ** 2)

  @staticmethod
  def p(V):
    return V / np.sum(V)

  @staticmethod
  def sfd(dose):
    """Eq.1: Fraction of viable cancer cells that survive after radiation dose
    return np.exp(

      -self.alpha*dose - self.beta * self.G(self.repair_rate * delivery_time) * (dose**2)
    )
    """
    return {20: 0.265, 8: 0.664, 6: 0.783}[dose]

  @staticmethod
  def aid(dose):
    """Eq.2:Fraction of cells that will undergo immunogenic cell death after
    radiation dose

    return min(1, (self.gamma/dose) * np.exp(-((np.log(dose) - self.mu)**2)/self.rho))
    """
    return {20: 0.194, 8: 0.984, 6: 0.367}[dose]

  def bed(self, dose, n):
    """ Biologicaly effective RT dose
    """
    return n * dose * (1 + dose / self.theta)

  def C_dot(self, C, D, I, VE, r, k, a):
    """Eq.4: Growth of viable cancer cells volume
    """
    V = C + D + I
    E = VE/V
    return r * C * (1 - V / k) - a * C * E

  def D_dot(self, C, D, I, VE, r, k, a, d):
    """Eq.5: CCs dying in a non-immunogenic manner
    """
    V = C + D + I
    E = VE/V
    return r * C * V / k + a *C * E - d * D

  def I_dot(self, I, d):
    """Eq.7:
    """
    return np.array([-d * I[0], 0])

  def VE_dot(self, C, D, I, VE, l, e, clr, omega, omega2, it, t):
    V = C + D + I
    u = self.u(t, it, e, clr)
    p = self.p(V)
    return - l * VE + (1 + u) * p * omega * np.sum(V + omega2 * I)

  def u(self, t, it, e, clr):
    administered_doses_time = it[it < t]
    return e * np.sum(np.exp(-clr * (t - administered_doses_time)))

  def ode(self, y, t, it, parameters, group):
    if 'TSA' in group:
      r = parameters[self.r]
      k = parameters[self.k]
      a = parameters[self.a]
      d = parameters[self.d]
      omega = parameters[self.omega]
    elif 'MCA' in group:
      r = parameters[self.r_MCA]
      k = parameters[self.k_MCA]
      a = parameters[self.a_MCA]
      d = parameters[self.d_MCA]
      omega = parameters[self.omega_MCA]
    l = parameters[self.l]
    clr = parameters[self.clr]
    omega2 = parameters[self.omega2]
    e = parameters[self.e]

    C, D, I, VE = np.split(y, 4, axis=0)

    return np.concatenate(
      [
        self.C_dot(C, D, I, VE, r, k, a),
        self.D_dot(C, D, I, VE, r, k, a, d),
        self.I_dot(I, d),
        self.VE_dot(C, D, I, VE, l, e, clr, omega, omega2, it, t)
      ],
      axis=0
    )

  def model(self, it, rt, initial_condition, parameters, group, x=None):
    # Eq.6: Introducing non-continous change to variables due to radiation.
    # ODEs are integrated on daily pieces
    if 'TSA' in group:
      dead_v = initial_condition[:2] * parameters[self.D_prop]
      VE = initial_condition[:2] * parameters[self.E]
    elif 'MCA' in group:
      dead_v = initial_condition[:2] * parameters[self.D_prop_MCA]
      VE = initial_condition[:2] * parameters[self.E_MCA]

    initial_condition += [-dead_v[0], -dead_v[1], dead_v[0], dead_v[1], 0, 0, VE[0], VE[1]]
    x = x if x is not None else self.x_data
    rt = dict(rt)
    results = list()

    for day in self.x:
      if day in rt.keys():
        rt_dose = rt[day]
        sfd = self.sfd(rt_dose)
        aid = self.aid(rt_dose)
        C = initial_condition[0]
        initial_condition[0] = C * sfd  # C1
        initial_condition[2] = initial_condition[2] + C * (1 - sfd) * (1 - aid)  # D1
        initial_condition[4] = initial_condition[4] + C * (1 - sfd) * aid  # E1
      _, initial_condition = integrate.odeint(
        self.ode, initial_condition, [day, day+1], args=(it, parameters, group) #, hmin=1e-7
      )
      if day in x:
        results.append(initial_condition.copy())

    return np.array(results)

  def model_dense(self, it, rt, initial_condition, parameters, group):
    # Eq.6: Introducing non-continous change to variables due to radiation.
    # ODEs are integrated on daily pieces
    if 'TSA' in group:
      dead_v = initial_condition[:2] * parameters[self.D_prop]
      VE = initial_condition[:2] * parameters[self.E]
    elif 'MCA' in group:
      dead_v = initial_condition[:2] * parameters[self.D_prop_MCA]
      VE = initial_condition[:2] * parameters[self.E_MCA]

    initial_condition += [-dead_v[0], -dead_v[1], dead_v[0], dead_v[1], 0, 0, VE[0], VE[1]]

    rt = dict(rt)
    # print(initial_condition)
    results = list()
    times = list()
    for day in self.x[:-1]:
      t = np.linspace(day, day + 1)
      if day in rt.keys():
        rt_dose = rt[day]
        sfd = self.sfd(rt_dose)
        aid = self.aid(rt_dose)
        C = initial_condition[0]
        initial_condition[0] = C * sfd  # C1
        initial_condition[2] = initial_condition[2] + C * (1 - sfd) * (1 - aid)  # D1
        initial_condition[4] = initial_condition[4] + C * (1 - sfd) * aid  # E1
      *r, initial_condition = integrate.odeint(
        self.ode, initial_condition, t, args=(it, parameters, group)
      )
      results.append(r)
      times.append(t[:-1])

    return np.concatenate(results, axis=0), np.concatenate(times, axis=0)


  def prediction(self, group, initial_condition, parameters):
    it = experimental_data[group]['it']
    rt = experimental_data[group]['rt']
    predictions = self.model(it, rt, initial_condition, parameters, group=group, x=self.x_data)

    V1 = np.expand_dims(np.sum(predictions[:, (0, 2, 4)], axis=1), axis=1)
    V2 = np.expand_dims(np.sum(predictions[:, (1, 3, 5)], axis=1), axis=1)
    return np.concatenate((V1, V2), axis=1)

  def plot(self, group, initial_condition, parameters):
    it = experimental_data[group]['it']
    rt = experimental_data[group]['rt']

    predictions, times = self.model_dense(it, rt, initial_condition, parameters, group)
    V1 = predictions[:, (0, 2, 4)]
    V2 = predictions[:, (1, 3, 5)]

    return V1, V2, times

  def err(self, parameters):
    V_error = np.zeros(shape=[len(self.experimental_data), len(self.x_data), 2])
    for i, group in enumerate(self.experimental_data):
      #print(group)
      V_data = experimental_data[group]["v"]
      initial_condition = np.array([V_data[0, 0], V_data[0, 1], 0., 0., 0., 0., 0., 0.])
      V = self.prediction(group, initial_condition, parameters)
      V_error[i, :, :] = (V / V_data - 1)**2

    return np.sum(V_error)
    
  def optimize(self, parameters_init):
    return optimize.minimize(
      self.err,
      parameters_init,
      #method='trust-constr',
      #options={'disp': True},
      #callback=print,
      bounds=[(0, 5 * param) for param in parameters_init],
    )


if __name__ == "__main__":
  lowest_err = 100
  for _ in range(100):
    ode = Parameterize_ODE()
    parameters = [np.random.uniform(0.7, 1.4) * p for p in start_params.copy()]
    x = ode.optimize(parameters)
    print(x)
    estimated_params = fited_values # x.x
    err = 1 #x.loss
    with open('results.txt', 'a+') as file_:
      file_.write('\nest:\n')
      file_.write(str(estimated_params))
      file_.write('\nerr:' + str(err))
      file_.write('\n')

      print('est:', estimated_params)
      print('err:', err)
      print()

    if err < lowest_err:
      print('plot...')
      for group in experimental_data:
        V_data = experimental_data[group]["v"]
        initial_condition = np.array([V_data[0,0], V_data[0,1], 0., 0., 0., 0., 0., 0.])
        l, r, times = ode.plot(
          group=group,
          initial_condition=initial_condition,
          parameters=estimated_params
        )
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(times, l[:, 0] + l[:, 1] + l[:, 2], lw=2, label='C', color='blue')
        ax1.plot(times, l[:, 1] + l[:, 2], lw=2, label='D', color='yellow')
        ax1.plot(times, l[:, 2], lw=2, label='I', color='red')
        ax1.plot(ode.x_data, V_data[:, 0], 'bo')
        ax1.grid()
        ax1.set_ylabel('left')
        ax1.set_ylim(0, 800 if 'TSA' in group else 1500)

        ax2.plot(times, r[:, 0] + r[:, 1] + r[:, 2], lw=2, label='C', color='blue')
        ax2.plot(times, r[:, 1] + r[:, 2], lw=2, label='D', color='yellow')
        ax2.plot(times, r[:, 2], lw=2, label='I', color='red')
        ax2.plot(ode.x_data, V_data[:, 1], 'bo')
        ax2.grid()
        ax2.set_ylabel('right')
        fig.suptitle(group.replace('_', ' '))
        ax2.set_ylim(0, 800 if 'TSA' in group else 1500)
        plt.savefig('plots/{}.png'.format(group))
