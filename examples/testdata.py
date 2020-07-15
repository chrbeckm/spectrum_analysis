import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

number_of_points = 1989

mu = [1290, 1300, 1312, 1350,
      1295, 1450, 1305, 1360,
      1292, 1303, 1310, 1340,
      1270, 1315, 1320, 1330]
sigma = [90, 100, 92, 105,
         96,  70, 95, 100,
         92, 102, 98, 102,
         94, 106, 94,  98]
intensity = 100000

testdir = 'testdata' + os.sep

if not os.path.exists(testdir):
    os.makedirs(testdir)

x = np.linspace(729, 1859, num=number_of_points)
y = []
for i, center in enumerate(mu):
    noise = np.random.randint(5,30,size=number_of_points)
    centernoise = np.random.randint(50,200)
    intensitynoise = np.random.randint(1,5)
    gaussian = intensity * stats.norm.pdf(x, center + 200, sigma[i]) + noise
    cauchy = (intensity / intensitynoise * 2 *
              stats.cauchy.pdf(x, center - 300 + centernoise, sigma[i]) + noise)
    temp = gaussian + cauchy
    y.append(temp)

for i, center in enumerate(mu):
    plt.plot(x, y[i], '.', markersize=1)
    np.savetxt(testdir + str(i + 1).zfill(4) + '.txt',
               np.column_stack([x, y[i]]))
plt.show()
