# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:05:03 2021

@author: dingxu
"""
import corner
import numpy as np


# Set up the parameters of the problem.
ndim, nsamples = 5, 50000

# Generate some fake data.
np.random.seed(42)
data1 = np.random.randn(ndim * 4 * nsamples // 5).reshape([4 * nsamples // 5, ndim])
data2 = (4*np.random.rand(ndim)[None, :] + np.random.randn(ndim * nsamples // 5).reshape([nsamples // 5, ndim]))
data = np.vstack([data1, data2])

# Plot it.
#figure = corner.corner(data2, labels=[r"$x$", r"$y$", r"$\log \alpha$", r"$\Gamma \, [\mathrm{parsec}]$"],
#                       quantiles=[0.16, 0.5, 0.84],
#                       show_titles=True, title_kwargs={"fontsize": 12})

figure = corner.corner(data2, labels=[r"$x$", r"$y$", r"$\log \alpha$", r"$\Gamma \, [\mathrm{parsec}]$"],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})