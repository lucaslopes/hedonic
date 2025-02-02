import pickle

pth = '/Users/lucas/Databases/Hedonic/PHYSA_2000/networks/2C_2000N/P_in = 0.05/Difficulty = 0.50/network_008.pkl'
paths = [pth]

for pth in paths:
    with open(pth, 'rb') as f:
        g = pickle.load(f)
    V = g.vcount()
    membership = [0 if i < int(V/2) else 1 for i in range(V)]
    res = g.density()
    fractions = g.resolution_spectrum(membership, [res])