import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# Initial state (on equator)
psi0 = (basis(2,0) + basis(2,1)).unit()

# Time array
tlist = np.linspace(0, 10, 200)

# Relaxation rate (1/T1)
gamma = 0.5

# Collapse operator for T1 (decay |1> -> |0>)
c_ops = [np.sqrt(gamma) * sigmam()]

# No Hamiltonian (just relaxation)
H = 0 * sigmaz()
result = mesolve(H, psi0, tlist, c_ops, [])

sx = expect(sigmax(), result.states)
sy = expect(sigmay(), result.states)
sz = expect(sigmaz(), result.states)

sz = -sz  # Invert sz for better visualization (|0> is up, |1> is down)

#copy frames
from qutip import Bloch
import imageio.v2 as imageio

frames = []
filenames = []

for i in range(len(tlist)):
    b = Bloch()
    b.add_vectors([sx[i], sy[i], sz[i]])
    
    filename = f"frame_{i}.png"
    b.save(filename)

    filenames.append(filename)                 # store filename
    frames.append(imageio.imread(filename))    # store image

imageio.mimsave(r"C:\Users\AnastasiaVelentza\Downloads\t1_relaxation.gif", frames, fps=50, loop=0)


#delete the individual frames
import os

for f in filenames:
    os.remove(f)