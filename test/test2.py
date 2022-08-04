import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

fig = plt.figure()

fig.set_figheight(5)
fig.set_figwidth(10)

# Adds subplot on position 1
ax = fig.add_subplot(121)
# Adds subplot on position 2
ax2 = fig.add_subplot(122)

ax.plot(x, y)
ax2.plot(x, z)
plt.show()
        