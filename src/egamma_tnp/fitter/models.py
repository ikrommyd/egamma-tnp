from __future__ import annotations

from numba_stats import crystalball_ex, expon

from egamma_tnp.fitter.basepdf import BasePDF


# Define DoubleCrystalBall PDF
class DoubleCrystalBall(BasePDF):
    def _unnormalized_pdf(self, x, mu, sigma, alphal, nl, alphar, nr):
        return crystalball_ex.pdf(x, alphal, nl, sigma, alphar, nr, sigma, mu)

    def _unnormalized_cdf(self, x, mu, sigma, alphal, nl, alphar, nr):
        return crystalball_ex.cdf(x, alphal, nl, sigma, alphar, nr, sigma, mu)


# Define Exponential PDF
class Exponential(BasePDF):
    def _unnormalized_pdf(self, x, lambd):
        return expon.pdf(x, self.norm_range[0], lambd)

    def _unnormalized_cdf(self, x, lambd):
        return expon.cdf(x, self.norm_range[0], lambd)
