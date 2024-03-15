import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from cbx.objectives import Rosenbrock
fig = plt.figure(figsize=(15,5))
x_min = -2.
x_max = 2.
y_min = -1.
y_max = 3.
f = Rosenbrock()

num_pts_landscape = 150
xx = np.linspace(x_min, x_max, num_pts_landscape)
yy = np.linspace(y_min, y_max, num_pts_landscape)
XX, YY = np.meshgrid(xx,yy)
XXYY = np.stack((XX.T,YY.T)).T
Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
Z[:,:,0:2] = XXYY
ZZ = f(Z)

ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122, projection='3d')
cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
ax0.contour(cs, colors='orange', alpha=0.2)
ax0.plot(1,1, color='orange', marker='x', markersize=10)
ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
ax0.set_title('Contour plot')
ax1.set_title('Surface plot')