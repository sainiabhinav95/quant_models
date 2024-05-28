import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

class HullWhiteModel:
    def __init__(self, yield_curve: ql.YieldTermStructureHandle, analysis_date: ql.Date,
                 time_step: int = 360, projection_tenor: int = 30, num_paths: int = 4096, params: list = [0.01, 0.01]) -> None:
        # Initial Setup
        self.as_of_date = analysis_date
        self.yield_curve = yield_curve
        ql.Settings.instance().evaluationDate = self.as_of_date

        # Hull White Model variables
        self.timestep = time_step
        self.length = projection_tenor # in years
        self.num_paths = num_paths
        self.sigma = params[0]
        self.a = params[1]
        

        # Conventions
        self.day_count = ql.Thirty360(ql.Thirty360.BondBasis)

        # Default Calibration
        self._calibrate()

    def _calibrate(self):
        print("Calibrating Hull White Model")
        hw_process = ql.HullWhiteProcess(self.yield_curve, a=self.a, sigma=self.sigma)

        # Generate random numbers
        self.rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(self.timestep, ql.UniformRandomGenerator()))

        # Generate discretised random paths
        self.seq = ql.GaussianPathGenerator(hw_process, self.length, self.timestep, self.rng, False)
        print("Calibrated!")

    def set_params(self, a: float, sigma: float):
        self.a = a
        self.sigma = sigma

        # Recalibrate the model
        self._calibrate()
    
    # We define alpha as per above equations
    def get_alpha(self, time: float):
        # YF for 1-day (proxy for short rate)
        yf = self.day_count.yearFraction(self.as_of_date, self.as_of_date + ql.Period(str("1D")))
        forward = self.yield_curve.forwardRate(time, time + yf,
                                               ql.Compounded).rate()
        alpha = forward + 0.5* np.power(self.sigma/self.a*(1.0 - np.exp(-self.a*time)), 2)
        return alpha

    def generate_paths(self):
        arr = np.zeros((self.num_paths, self.timestep+1))
        for i in range(self.num_paths):
            sample_path = self.seq.next()
            path = sample_path.value()
            time = [path.time(j) for j in range(len(path))]
            value = [path[j] for j in range(len(path))]
            arr[i, :] = np.array(value)
        return np.array(time), arr
 
 