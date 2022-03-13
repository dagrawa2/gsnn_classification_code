
def group(group):
	letter, degree = tuple(group.split("_"))
	degree = int(degree)
	generators = group_dict[letter+"_n"](degree)
	return generators


def cyclic(n):
	rho = list(range(2, n+2))
	rho[-1] = 1
	return [rho]

def dihedral(n):
	rho = list(range(2, n+2))
	rho[-1] = 1
	tau = list(range(1, n+1))
	tau.reverse()
	return [rho, tau]

def symmetric(n):
	rho = list(range(2, n+2))
	rho[-1] = 1
	tau = list(range(1, n+1))
	tau[0] = 2
	tau[1] = 1
	return [rho, tau]


group_dict = {
	"C_n": cyclic, 
	"D_n": dihedral, 
	"S_n": symmetric, 
	"Z_n": cyclic
}
