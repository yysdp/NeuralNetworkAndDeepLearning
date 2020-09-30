import numpy as np
import matplotlib.pyplot as plt
#%matplotlib qt
fashion = lambda x:(x**3)-3*(x**2)+7
x=np.linspace(-1,3,100)
plt.plot(x,fashion(x),'r-')
plt.show()

a = np.random.rand(3,3)
print(a-0.5)