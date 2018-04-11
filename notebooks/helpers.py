# Standard library
from os import path
import glob
import re

# Third-party
import astropy.units as u
from astropy.constants import G
from astropy.table import QTable
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GaussianMixture

# Custom
import mesa_reader as mr
from twoface.mass import period_at_surface


##############################################################################
# MESA

class MESAHelper:

    def __init__(self, mesa_path):
        self._LOG_PATHS = sorted(glob.glob(path.join(mesa_path, '*Msun/LOGS')))

        self.path_dict = {}
        pattr = re.compile('\/([0-9\.]+)Msun')
        for PATH in self._LOG_PATHS:
            mass = pattr.search(PATH).groups()[0]
            self.path_dict[mass] = PATH

        self.load_all_history()

    def get_closest_key(self, mass):
        idx = np.abs(mass.to(u.Msun).value - self._h_M).argmin()
        return self._h_keys[idx]

    def get_closest_history(self, mass):
        key = self.get_closest_key(mass)
        return self.h[key]

    def load_all_history(self):
        self.h = {}
        for k, PATH in self.path_dict.items():
            self.h[k] = mr.MesaData(path.join(PATH, 'history.data'))

        self._h_keys = np.array([k for k in self.h.keys()])
        self._h_M = np.array([float(k) for k in self._h_keys])

        idx = self._h_M.argsort()
        self._h_keys = self._h_keys[idx]
        self._h_M = self._h_M[idx]


##############################################################################
# Population synthesis

class MatchedSimulatedSample:

    def __init__(self, logg, M1, M2, mesa_helper, seed=42):
        self.rnd = np.random.RandomState(seed=seed)

        # Use a KDE to estimate the M1 distribution:
        _m1 = M1[np.isfinite(M1)].to(u.Msun).value
        self._m1_kde = KernelDensity(bandwidth=0.4, kernel='epanechnikov')
        self._m1_kde.fit(_m1.reshape(-1, 1))

        # Use a GMM to estimate the logg distribution:
        _logg = logg[logg > -999]
        self._logg_gmm = GaussianMixture(n_components=3, covariance_type='diag',
                                         random_state=self.rnd)
        self._logg_gmm.fit(_logg.reshape(-1, 1))

        self.mesa_helper = mesa_helper

    def generate_sample(self, size):
        t = QTable()

        # Fundamental properties
        t['logg'] = self.sample_logg(size)
        t['M1_orig'] = self.sample_m1(size=size)
        t['M2'] = self.sample_m2(t['M1_orig'])

        mu = 0.4
        sig = 0.3
        a = (0 - mu) / sig
        b = (1 - mu) / sig
        t['e'] = truncnorm.rvs(loc=mu, scale=sig, a=a, b=b, size=size)

        # Computed quantities
        g = 10**t['logg'] * u.cm/u.s**2
        t['R1'] = np.sqrt(G*t['M1_orig'] / g).to(u.Rsun)

        # Discretized M1
        m1s = []
        for i in range(size):
            m_idx = np.abs(self.mesa_helper._h_M -
                           t['M1_orig'].value[i]).argmin()
            m1s.append(self.mesa_helper._h_M[m_idx])
        t['M1'] = m1s * u.Msun

        P_min = period_at_surface(t['M1'], t['logg'], t['e'], M2=t['M2'])
        t['P'] = 10 ** self.rnd.uniform(np.log10(P_min.to(u.day).value),
                                        np.log10(8192), size) * u.day
        t['a'] = np.cbrt(t['P']**2/(2*np.pi)**2 *
                         G * (t['M1'] + t['M2'])).to(u.au)

        return t

    def sample_m1(self, size=1):
        x = self.rnd.uniform(0, 3.5, size=size*128)
        ykde = np.exp(self._m1_kde.score_samples(x.reshape(-1, 1)))
        y = self.rnd.uniform(0, ykde.max(), size=x.shape)
        x = x[y < ykde]
        if len(x) < size:
            raise RuntimeError()
        return x[:size] * u.Msun

    def sample_logg(self, size=1):
        return self._logg_gmm.sample(n_samples=size)[0][:, 0]

    def sample_q(self, size=1):
        logq = self.rnd.uniform(-2, 0, size=size)
        return 10**logq

    def sample_m2(self, M1):
        size = len(M1)
        return M1 * self.sample_q(size=size)


##############################################################################
# Binary orbit evolution

# I estimated these by eye: the index at which the red clump phase starts
rc = {}
rc['0.8'] = 16000
rc['1.0'] = 17700
rc['1.2'] = 17500
rc['1.4'] = 17400
rc['1.6'] = 17200
rc['1.8'] = 16000
rc['2.0'] = 11800
rc['2.5'] = 5000
rc['3.0'] = 5000

def tcirc_inv(L, Menv, R, M1, M2, a, f=1.):
    """Inverse of the circularization time in units of [1/Gyr]"""
    q = M2 / M1
    L_MenvR = np.cbrt(L / (Menv * R**2))
    return L_MenvR * Menv / M1 * q * (1+q) * (R / a)**8


def compute_dlne(logg, M1, M2, a, mesa_helper, f=1.):
    """Compute the change in eccentricity given binary properties.
    This function assumes the semi-major axis stays constant.
    """
    mesa = mesa_helper

    # Retrieve MESA model for model with closest mass
    h_key = mesa.get_closest_key(M1)
    h = mesa.get_closest_history(M1)

    # Evolve starting from MS:
    min_idx = h.log_R.argmin()
    logg_idx = np.abs(h.log_g[min_idx:rc[h_key]] - logg).argmin() + min_idx
    if (logg_idx - min_idx) <= 0:
        raise ValueError('Invalid logg {0} for {1} model'.format(logg, h_key))

    slc = slice(min_idx, logg_idx)
    L = h.L[slc] * u.Lsun
    Menv = h.cz_xm[slc] * u.Msun
    R = h.R[slc] * u.Rsun
    time = h.star_age[slc] * u.yr

    y = -tcirc_inv(L, Menv, R, M1, M2, a, f=f)
    dlne = simps(y.to(1/u.Gyr).value,
                 x=time.to(u.Gyr).value)

    return dlne


def dlne_dlna_func(y, t, log_Lfn, log_Menvfn, log_Rfn, M1, M2, f=1.):
    lne, lna = y

    a = np.exp(lna) * u.Rsun
    e = np.exp(lne)

    L = 10**log_Lfn(t) * u.Lsun
    Menv = 10**log_Menvfn(t) * u.Msun
    R = 10**log_Rfn(t) * u.Rsun

    fac = -tcirc_inv(L, Menv, R, M1, M2, a, f=f)
    dlna_dt = -7.2 * f * fac * e**2
    dlne_dt = -f * fac

    return np.array([dlne_dt, dlna_dt])


def euler(f, y0, t0, dt, nsteps, args=None):
    if args is None:
        args = ()

    ys = []
    ts = []

    t = t0
    y = np.atleast_1d(y0)
    ts.append(t)
    ys.append(y)

    for i in range(nsteps):
        y = y + dt * f(y, t, *args)
        t = t + dt
        ys.append(y)
        ts.append(t)

    return np.array(ts), np.array(ys)


def solve_final_ea(e0, a0, logg, M1, M2, mesa_helper, nsteps=10000, f=1.):
    mesa = mesa_helper

    # Retrieve MESA model for model with closest mass
    h_key = mesa.get_closest_key(M1)
    h = mesa.get_closest_history(M1)

    # Evolve starting from MS:
    min_idx = h.log_R.argmin()
    logg_idx = np.abs(h.log_g[min_idx:rc[h_key]] - logg).argmin() + min_idx
    if (logg_idx - min_idx) <= 0:
        raise ValueError('Invalid logg {0} for {1} model'.format(logg, h_key))

    slc = slice(min_idx, logg_idx)
    L = h.L[slc] # Lsun
    Menv = h.cz_xm[slc] # Msun
    R = h.R[slc] # Rsun
    time = h.star_age[slc] # yr
    time_Gyr = time / 1E9 # Gyr

    dt = (time_Gyr[-1] - time_Gyr[0]) / nsteps
    y0 = np.array([np.log(e0), np.log(a0.to(u.Rsun).value)])

    Lfn = interp1d(time_Gyr, np.log10(L), kind='linear',
                   bounds_error=False)
    Rfn = interp1d(time_Gyr, np.log10(R), kind='linear',
                   bounds_error=False)
    Menvfn = interp1d(time_Gyr, np.log10(Menv), kind='linear',
                      bounds_error=False)
    t_y, y = euler(dlne_dlna_func, y0, time_Gyr[0], dt=dt, nsteps=nsteps,
                   args=(Lfn, Menvfn, Rfn, M1.value, M2.value))

    ef, af = np.exp(y[-1])
    return (ef, af * u.Rsun)
