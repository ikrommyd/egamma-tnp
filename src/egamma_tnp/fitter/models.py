from __future__ import annotations

from numba_stats import cmsshape, crystalball_ex, expon, norm

from egamma_tnp.fitter.basepdf import BasePDF, register_pdf


# Define Normal (Gaussian) PDF
@register_pdf("Normal")
class Normal(BasePDF):
    def _unnormalized_pdf(self, x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    def _unnormalized_cdf(self, x, mu, sigma):
        return norm.cdf(x, mu, sigma)


# Define DoubleCrystalBall PDF
@register_pdf("DoubleCrystalBall")
class DoubleCrystalBall(BasePDF):
    def _unnormalized_pdf(self, x, mu, sigma, alphal, nl, alphar, nr):
        return crystalball_ex.pdf(x, alphal, nl, sigma, alphar, nr, sigma, mu)

    def _unnormalized_cdf(self, x, mu, sigma, alphal, nl, alphar, nr):
        return crystalball_ex.cdf(x, alphal, nl, sigma, alphar, nr, sigma, mu)


# Define Exponential PDF
@register_pdf("Exponential")
class Exponential(BasePDF):
    def _unnormalized_pdf(self, x, lambd):
        return expon.pdf(x, self.norm_range[0], lambd)

    def _unnormalized_cdf(self, x, lambd):
        return expon.cdf(x, self.norm_range[0], lambd)


# Define CMSShape PDF
@register_pdf("CMSShape")
class CMSShape(BasePDF):
    def _unnormalized_pdf(self, x, beta, gamma, loc):
        return cmsshape.pdf(x, beta, gamma, loc)

    def _unnormalized_cdf(self, x, beta, gamma, loc):
        return cmsshape.cdf(x, beta, gamma, loc)
