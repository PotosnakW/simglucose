from simglucose.simulation.scenario import Action, Scenario
import numpy as np
from scipy.stats import truncnorm
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RandomScenario(Scenario):
    def __init__(self, start_time, patient, seed=None):
        Scenario.__init__(self, start_time=start_time)
        self.patient = patient
        self.seed = seed
        
    def get_action(self, t):
        # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        if t_sec < 1:
            logger.info('Creating new one day scenario ...')
            self.scenario = self.create_scenario()

        t_min = np.floor(t_sec / 60.0)

        if t_min in self.scenario['meal']['time']:
            logger.info('Time for meal!')
            idx = self.scenario['meal']['time'].index(t_min)
            return Action(meal=self.scenario['meal']['amount'][idx])
        else:
            return Action(meal=0)

    def create_scenario(self):
        scenario = {'meal': {'time': [], 'amount': []}}

        # # Probability of taking each meal
        # # [breakfast, snack1, lunch, snack2, dinner, snack3]
        # prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        # time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60
        # time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60
        # time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        # time_sigma = np.array([60, 30, 60, 30, 60, 30])
        # amount_mu = [45, 10, 70, 10, 80, 10]
        # amount_sigma = [10, 5, 10, 5, 10, 5]

        if self.patient._params.Age < 13:
            height = 140
        elif (self.patient._params.Age >= 13) & (self.patient._params.Age < 20):
            height = 170
        elif self.patient._params.Age > 20:
            height = 175
        
        bmr = 66.5 + 13.75*self.patient._params.BW + 5.003*height - \
              6.755*self.patient._params.Age 
        bmr = bmr * 1.2 # 1.2 implies sedetary patient
        expected_carbs = (bmr*0.45)/4 # 45% of calories assumed from carbs, 4 calories per carb

        # Probability of taking each meal
        # [breakfast, lunch, dinner]
        prob = [0.95, 0.95, 0.95]
        time_lb = np.array([5, 10, 16]) * 60
        time_ub = np.array([9, 14, 20]) * 60
        time_mu = np.array([7, 12, 18]) * 60
        time_sigma = np.array([1, 1, 1]) * 60
        amount_mu = np.array([0.285, 0.330, 0.385])*expected_carbs 
        amount_sigma = amount_mu*0.15

        for p, tlb, tub, tbar, tsd, mbar, msd in zip(prob, time_lb, time_ub,
                                                     time_mu, time_sigma,
                                                     amount_mu, amount_sigma):
            if self.random_gen.rand() < p:
                tmeal = np.round(
                    truncnorm.rvs(a=(tlb - tbar) / tsd,
                                  b=(tub - tbar) / tsd,
                                  loc=tbar,
                                  scale=tsd,
                                  random_state=self.random_gen))
                scenario['meal']['time'].append(tmeal)
                scenario['meal']['amount'].append(
                    max(round(self.random_gen.normal(mbar, msd)), 0))

        return scenario

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()


if __name__ == '__main__':
    from datetime import time
    from datetime import timedelta
    import copy
    now = datetime.now()
    t0 = datetime.combine(now.date(), time(6, 0, 0, 0))
    t = copy.deepcopy(t0)
    sim_time = timedelta(days=2)

    scenario = RandomScenario(seed=1)
    m = []
    T = []
    while t < t0 + sim_time:
        action = scenario.get_action(t)
        m.append(action.meal)
        T.append(t)
        t += timedelta(minutes=1)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.plot(T, m)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    plt.show()
