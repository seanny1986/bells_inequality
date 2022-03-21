# Copyright (c) 2021 Sean Morrison
# See https://arxiv.org/pdf/1507.00106.pdf by Richard Gill for additional reading.

import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
 
def rotate(xy, theta):
    R = np.array([[cos(theta), -sin(theta)],
                  [sin(theta),  cos(theta)]])
    return R.dot(xy)
 
def project(u, v):
    return np.dot(u, v)
 
def generate_samples(n=10000):
 
    # uniformly sample a 2D vector from the unit circle
    angles = np.random.uniform(low=0, high=2 * pi, size=(n,1))
    ps = 2 / np.sqrt(np.random.uniform(low=1, high=4, size=(n,))) - 1   # Pearle, 1970
    xs = np.cos(angles)
    ys = np.sin(angles)
    return np.hstack([xs, ys]), ps
 
# number of samples
n = 10000
 
# generate entangled particles. I've called these xy and uv
xy, ps = generate_samples(n)
uv = -xy
 
# P, Q, R are measurement settings. In this case, we will take 3 axes, and measure
# if the generated vector is positive or negative along the axis by projecting it,
# and seeing if the projection is positive or negative
P = np.array([1., 0.])
R = rotate(P, 240 * pi / 180)
 
# we will vary the angle of Q, and use the inequality Pr(P+,Q+) \leq Pr(P+,R+) + Pr(Q+,R+) to
# ensure that Bell's inequality holds over the test range
angle_settings = np.linspace(0, 2 * pi)
PQ_data, PR_data, QR_data = [], [], []
correlation = []
correlation_loophole = []
for phi in angle_settings:
 
    # rotate the sensor axis Q
    Q = rotate(P, phi)
 
    # project the data point onto detectors P, Q, R
    projected_p1 = np.array([project(x, P) for x in xy])
    projected_q1 = np.array([project(x, Q) for x in xy])
    projected_r1 = np.array([project(x, R) for x in xy])
 
    # project the entangled data point onto detectors P, Q, R
    projected_p2 = np.array([project(x, P) for x in uv])
    projected_q2 = np.array([project(x, Q) for x in uv])
    projected_r2 = np.array([project(x, R) for x in uv])
 
    # get Pr(P+,Q+), Pr(P+,R+), Pr(Q+,R+)
    PQ_positive = np.logical_and(projected_p1 > 0, projected_q2 > 0)
    PR_positive = np.logical_and(projected_p1 > 0, projected_r2 > 0)
    QR_positive = np.logical_and(projected_r1 > 0, projected_q2 > 0)
    PQ_data.append(np.sum(PQ_positive) / n)
    PR_data.append(np.sum(PR_positive) / n)
    QR_data.append(np.sum(QR_positive) / n)
 
    # get quantum correlation. This is just (# of agreements - # of disagreements) / n between detectors
    # P and Q. Why? Because if the detectors always agree, the correlation is 1. If they always disagree,
    # the correlation is -1. If they disagree as much as they agree, the correlation is zero.
    agreement = np.sum(PQ_positive + np.logical_and(projected_p1 < 0, projected_q2 < 0))
    disagreement = np.sum(np.logical_and(projected_p1 > 0, projected_q2 < 0) + np.logical_and(projected_p1 < 0, projected_q2 > 0))
    corr = (agreement - disagreement) / n
    correlation.append(corr)

    # we'll use the detection loophole to generate the desired quantum correlation function. This boils down
    # using rejection sampling, with the probability of rejection being p = 2 / sqrt(V) - 1 for V ~ U(1, 4).
    idxs_p1 = np.where(np.abs(projected_p1) > ps)
    idxs_q2 = np.where(np.abs(projected_q2) > ps)
    idxs = np.intersect1d(idxs_p1, idxs_q2)

    # only calculate agreement and disagreement using non-rejected indexes
    agreement = np.sum(PQ_positive[idxs] + np.logical_and(projected_p1[idxs] < 0, projected_q2[idxs] < 0))
    disagreement = np.sum(np.logical_and(projected_p1[idxs] > 0, projected_q2[idxs] < 0) + np.logical_and(projected_p1[idxs] < 0, projected_q2[idxs] > 0))
    corr = (agreement - disagreement) / len(idxs)
    correlation_loophole.append(corr)

# Bell's inequality holds, and the correlation plot is linear. Shit.
plt.plot(angle_settings * 180 / pi, PQ_data)
plt.plot(angle_settings * 180 / pi, [p + q for p, q in zip(PR_data, QR_data)])
plt.plot(angle_settings * 180 / pi, correlation)
plt.plot(angle_settings * 180 / pi, correlation_loophole)
plt.plot(angle_settings * 180 / pi, -np.cos(angle_settings), "--k")
plt.title("Bell's inequality with locally hidden variables")
plt.xlim([0,360])
plt.ylim([-1, 1])
plt.ylabel("Pr(.)")
plt.xlabel("angle between P, Q (degrees)")
plt.legend(["Pr(P > 0, Q > 0)","Pr(P > 0, R > 0) + Pr(R > 0, Q > 0)", "correlation", "correlation_loophole", "QM prediction"])
plt.show()
