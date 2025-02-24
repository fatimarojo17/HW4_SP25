import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

def ln_PDF(D, mu, sig):
    if D == 0.0:
        return 0.0
    p = 1/(D*sig*math.sqrt(2*math.pi))
    _exp = -((math.log(D)-mu)**2)/(2*sig**2)
    return p*math.exp(_exp)

def tln_PDF(D, mu, sig, F_DMin, F_DMax):
    return ln_PDF(D, mu, sig) / (F_DMax - F_DMin)

def F_tlnpdf(D, mu, sig, D_Min, D_Max, F_DMax, F_DMin):
    if D > D_Max or D < D_Min:
        return 0
    P, _ = quad(lambda x: tln_PDF(x, mu, sig, F_DMin, F_DMax), D_Min, D)
    return P

def makeSample(mu, sig, D_Min, D_Max, F_DMax, F_DMin, N=100):
    probs = np.random.rand(N)
    d_s = [fsolve(lambda D: F_tlnpdf(D, mu, sig, D_Min, D_Max, F_DMax, F_DMin) - p, D_Min)[0] for p in probs]
    return d_s

def sampleStats(D, doPrint=False):
    mean = np.mean(D)
    var = np.var(D, ddof=1)
    if doPrint:
        print(f"mean = {mean:.3f}, variance = {var:.3f}")
    return mean, var

def getFDMaxFDMin(mu, sig, D_Min, D_Max):
    F_DMax, _ = quad(lambda x: ln_PDF(x, mu, sig), 0, D_Max)
    F_DMin, _ = quad(lambda x: ln_PDF(x, mu, sig), 0, D_Min)
    return F_DMin, F_DMax

def makeSamples(mu, sig, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples):
    Samples = []
    Means = []
    Variances = []
    for _ in range(N_samples):
        sample = makeSample(mu, sig, D_Min, D_Max, F_DMax, F_DMin, N=N_sampleSize)
        mean, var = sampleStats(sample)
        Samples.append(sample)
        Means.append(mean)
        Variances.append(var)
        print(f"Sample {_}: mean = {mean:.3f}, variance = {var:.3f}")
    return Samples, Means, Variances

def main():
    mu = math.log(2)
    sig = 1
    D_Max = 1
    D_Min = 3.0/8.0
    N_samples = 11
    N_sampleSize = 100

    F_DMin, F_DMax = getFDMaxFDMin(mu, sig, D_Min, D_Max)
    Samples, Means, Variances = makeSamples(mu, sig, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples)
    stats_of_Means = sampleStats(Means)
    print(f"Mean of the sampling mean: {stats_of_Means[0]:.3f}")
    print(f"Variance of the sampling mean: {stats_of_Means[1]:.6f}")

if __name__ == '__main__':
    main()
