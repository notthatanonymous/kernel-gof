import kgof
import kgof.data as data
import kgof.density as density
import kgof.goftest as gof
import kgof.kernel as kernel
import kgof.plot as plot
import kgof.util as util

import matplotlib
#import matplotlib.pyplot as plt
import autograd.numpy as np
# import scipy.stats as stats

def prob2d_pqgauss(qmean=np.array([0, 0]), QVar=np.eye(2), seed=2):
    """
    Construct a problem where p = N(0, I), q = N(0, QVar).
    
    Return p, a DataSource (for q)
    """
    p = density.Normal(np.array([0, 0]), np.eye(2))
    ds = data.DSNormal(qmean, QVar)
    return p, ds


def func_fssd_power_criterion(p, X, k, V):
    """
    Return the value of the power criterion of FSSD.
    p: model density
    X: n x d data matrix
    k: a Kernle
    V: J x d numpy array. J test locations
    """
    dat = data.Data(X)
    return gof.FSSD.power_criterion(p, dat, k, V, reg=1e-5, use_unbiased=False)



def prob_1dmixture(pm=0, pv=1, qm1=0, qv1=1, qm2=1, qv2=1, seed=3):
    """
    A 1D problem where both p and q are 2-component Gaussian mixture models.
    p(x) = 0.5*N(0, 1) + 0.5*N(pm, pv)
    q(x) = 0.5*N(qm1, qv1) + 0.5*N(qm2, qv2)
    
    Return p and q (both are UnnormalizedDensity)
    """
    assert pv > 0
    assert qv1 > 0
    assert qv2 > 0
    p = density.IsoGaussianMixture(means=np.array([[0], [pm]]), 
                                  variances=np.array([1, pv]))
    q = density.IsoGaussianMixture(means=np.array([[qm1], [qm2]]),
                                  variances=np.array([qv1, qv2]))
    return p, q

def func_interactive_1dmixture(pm=0, pv=1, qm1=0, qv1=1, qm2=1, qv2=1):
    
    seed = 84
    p, q = prob_1dmixture(pm=pm, pv=pv, qm1=qm1, qv1=qv1, qm2=qm2,
                          qv2=qv2, seed=seed)
    
    # n = sample size to draw from q
    n = 600
    gwidth2 = 1.5**2
    # generate data from q
    ds = q.get_datasource()
    dat = ds.sample(n, seed=seed+3)
    Xs = dat.data()
    # kernel
    k = kernel.KGauss(sigma2=gwidth2)
    def score_function(Vs):
        """
        Vs: m x d test locations. 
        Evaluate the score at m locations
        """
        m = Vs.shape[0]
        objs = np.zeros(m)
        for i in range(m):
            v = Vs[i, :]
            obj = func_fssd_power_criterion(p, Xs, k, v[np.newaxis, :])
            objs[i] = obj
        return objs
    print('p = 0.5*N(0, 1) + 0.5*N({}, {})'.format(pm, pv))
    print('q = 0.5*N({}, {}) + 0.5*N({}, {})'.format(qm1, qv1, qm2, qv2))


# Assume two dimensions.
d = 2
def isogauss_log_den(X):
    """
    Evaluate the log density of the standard isotropic Gaussian 
    at the points (rows) in X.
    Note that the density is NOT normalized. 
    
    X: n x d nd-array
    return a length-n array
    """
    # d = dimension of the input space
    unden = -np.sum(X**2, 1)/2.0
    return unden

# p is an UnnormalizedDensity object
p = density.from_log_den(d, isogauss_log_den)

# Let's assume that m = 1.
m = 1

# Draw n points from q
seed = 5
np.random.seed(seed)
n = 300
X = np.random.randn(n, 2) + np.array([m, 0])


# dat will be fed to the test.
dat = data.Data(X)

# We will use some portion of the data for parameter tuning, and the rest for testing.
tr, te = dat.split_tr_te(tr_proportion=0.5, seed=2)

# J is the number of test locations (or features). Typically not larger than 10.
J = 1

# There are many options for the optimization. 
# Almost all of them have default values. 
# Here, we will list a few to give you a sense of what you can control.
# Full options can be found in gof.GaussFSSD.optimize_locs_widths(..)
opts = {
    'reg': 1e-2, # regularization parameter in the optimization objective
    'max_iter': 50, # maximum number of gradient ascent iterations
    'tol_fun':1e-4, # termination tolerance of the objective
}

# make sure to give tr (NOT te).
# do the optimization with the options in opts.
V_opt, gw_opt, opt_info = gof.GaussFSSD.optimize_auto_init(p, tr, J, **opts)

# alpha = significance level of the test
alpha = 0.01
fssd_opt = gof.GaussFSSD(p, gw_opt, V_opt, alpha)

# return a dictionary of testing results
test_result = fssd_opt.perform_test(te)
print(test_result)

