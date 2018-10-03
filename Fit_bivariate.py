import pickle
import sys
import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
def bivariate_gaussian((x1,x2),mu1,mu2,sigma1,sigma2,rho):
	A = 1 / (2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - rho ** 2))
	B = -1 / (2 * (1 - rho ** 2))
	X1 = (x1 - mu1) ** 2 / sigma1 ** 2
	X12 = 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
	X2 = (x2 - mu2) ** 2 / sigma2 ** 2
	return A* np.exp(B*(X1 - X12 + X2))
with open('/nfs/slac/g/exo/shaolei/denoising/pkls/EnergyInfo_'+ str(sys.argv[1]) +'_denoised_2804_to_9519_trim_copy.pkl') as energy:
	EnergyInfo = pickle.load(energy)
ce = np.array([event[0] for event in EnergyInfo if event[1] > 5000 - event[0] and event[1] < 5600 - event[0]])
se = np.array([event[1] for event in EnergyInfo if event[1] > 5000 - event[0] and event[1] < 5600 - event[0]])
xmin = 2200
ymin = 2200
xmax = 3000
ymax = 3000
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([ce, se])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
initial_guess = [2615,2615,50,50,0.1]
popt,pcov = curve_fit(bivariate_gaussian,(X.ravel(),Y.ravel()),Z.ravel(),p0=initial_guess)
z_cf = bivariate_gaussian((X,Y),*popt)
Title = r'$\mu_1 = $' + str(popt[0]) + r' $\mu_2 = $' + str(popt[1]) + r' $\sigma_1 = $' + str(popt[2]) + r'$\sigma_2 = $' + str(popt[3]) + r'$\rho = $' + str(popt[4])
#calculate the angle
sigmai = popt[2]
sigmas = popt[3]
I = popt[0]
S = popt[1]
rho = popt[4]
theta = np.arctan(-2 * sigmai * sigmas * rho / (sigmas**2 - sigmai**2)) / 2
R = (I * np.cos(theta) + S * np.sin(theta)) 
sigma_r = (np.cos(theta) * sigmai) ** 2 + (np.sin(theta) * sigmas) ** 2 + 2 * rho * np.sin(theta) * np.cos(theta) * sigmai * sigmas
resolution_r = np.sqrt(sigma_r) / R
print "Resolution",resolution_r
print "R_I",sigmai / I
print "R_S",sigmas / S

lx = np.linspace(xmin,xmax,100)
ly = np.tan(theta) * (lx - I) + S
fig, ax = plt.subplots()
ax.imshow(np.rot90(z_cf), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
ax.plot(ce, se, 'k.', markersize=2)
ax.plot(lx, ly, '-')
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.grid(True)
plt.title(Title)
plt.ylabel('scintillation energy (keV)')
plt.xlabel('ionization energy (keV)')
plt.show()
