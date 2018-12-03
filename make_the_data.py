import numpy as np

def make_data():
	x = np.random.uniform(0,100,300)
	y = np.random.uniform(0,100,300)
	z = np.random.uniform(0,100,300)
	pr = np.asarray([x[k]*np.exp(-0.5*y[k]**2) + 0.1*y[k]*np.log(z[k]) - x[k]**(1/4) for k in range(len(x))])
	return np.column_stack([x,y,z]), pr

df, target = make_data()


to_print = np.column_stack([target, df])
np.savetxt('synthetic_data.txt', to_print)
