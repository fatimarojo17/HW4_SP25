import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

def ln_PDF(D, mu, sig):
    if D == 0.0:
        return 0.0
    p = 1 / (D * sig * math.sqrt(2 * math.pi))
    _exp = -((math.log(D) - mu) ** 2) / (2 * sig ** 2)
    return p * math.exp(_exp)

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

def plotGraphs(mu, sig, D_Min, D_Max, F_DMax, F_DMin):
    D_values = np.linspace(D_Min, D_Max, 1000)
    pdf_values = [tln_PDF(D, mu, sig, F_DMin, F_DMax) for D in D_values]
    cdf_values = [F_tlnpdf(D, mu, sig, D_Min, D_Max, F_DMax, F_DMin) for D in D_values]

    D_target = D_Min + (D_Max - D_Min) * 0.75
    F_target = F_tlnpdf(D_target, mu, sig, D_Min, D_Max, F_DMax, F_DMin)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(D_values, pdf_values, 'b', label='tln-PDF')
    axs[0].fill_between(D_values[D_values <= D_target], pdf_values[:len(D_values[D_values <= D_target])], color='gray', alpha=0.5)
    axs[0].annotate(f"P(D<={D_target:.2f}) = {F_target:.2f}", xy=(D_target, tln_PDF(D_target, mu, sig, F_DMin, F_DMax)), xytext=(D_target - 0.1, 1.2), arrowprops=dict(arrowstyle='->'))
    axs[0].set_ylabel("f(D)")
    axs[0].legend()
    axs[0].text(0.4 * (D_Max + D_Min), max(pdf_values) * 0.9, r'$f(D) = \frac{1}{D \sigma \sqrt{2 \pi}} e^{-\frac{(\ln D - \mu)^2}{2 \sigma^2}}$', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    axs[1].plot(D_values, cdf_values, 'b')
    axs[1].plot(D_target, F_target, 'ro')
    axs[1].vlines(D_target, 0, F_target, colors='k', linestyles='dashed')
    axs[1].hlines(F_target, D_Min, D_target, colors='k', linestyles='dashed')
    axs[1].set_xlabel("x")
    axs[1].set_ylabel(r"$\Phi(x) = \int_{D_{min}}^x f(D) dD$")
    axs[1].text(0.4 * (D_Max + D_Min), max(cdf_values) * 0.9, r'$\Phi(D) = \int_{D_{min}}^D f(D) dD$', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

def main():
    mu = math.log(2)
    sig = 1
    D_Max = 1
    D_Min = 3.0 / 8.0

    F_DMin, F_DMax = getFDMaxFDMin(mu, sig, D_Min, D_Max)
    plotGraphs(mu, sig, D_Min, D_Max, F_DMax, F_DMin)

if __name__ == '__main__':
    main()
