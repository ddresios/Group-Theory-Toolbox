import numpy as np

class PointGroup:
    def __init__(self, name, classes, n_ops, irreps):
        """
        classes: list of class labels
        n_ops: number of operations in each class
        irreps: dict of irreducible representations {name: character list}
        """
        self.name = name
        self.classes = classes
        self.n_ops = np.array(n_ops)
        self.irreps = irreps
        self.h = sum(n_ops)  # group order

    def reduce_representation(self, gamma_red):
        gamma_red = np.array(gamma_red)
        results = {}

        for irrep_name, chars in self.irreps.items():
            chars = np.array(chars)
            ai = (1 / self.h) * np.sum(self.n_ops * gamma_red * chars)
            results[irrep_name] = round(ai)

        return results


# -------------------------
#Point Groups 
# -------------------------
C4v = PointGroup(
    name="C4v",
    classes=["E", "2C4", "C2", "2σv", "2σd"],
    n_ops=[1, 2, 1, 2, 2],
    irreps={
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "B1": [1, -1, 1, 1, -1],
        "B2": [1, -1, 1, -1, 1],
        "E":  [2, 0, -2, 0, 0],
    }
)

C3v = PointGroup(
    name="C3v",
    classes=["E", "2C3", "3σv"],
    n_ops=[1, 2, 3],
    irreps={
        "A1": [1, 1, 1],
        "A2": [1, 1, -1],
        "E":  [2, -1, 0],
    }
)

C2v = PointGroup(
    name="C2v",
    classes=["E", "C2", "σv(xz)", "σv(yz)"],
    n_ops=[1, 1, 1, 1],
    irreps={
        "A1": [1, 1, 1, 1],
        "A2": [1, 1, -1, -1],
        "B1": [1, -1, 1, -1],
        "B2": [1, -1, -1, 1],
    }
)

Oh = PointGroup(
    name="Oh",
    classes=["E", "8C3", "6C2", "6C4", "3C2'", "i", "6S4", "8S6", "3σh", "6σd"],
    n_ops=[1, 8, 6, 6, 3, 1, 6, 8, 3, 6],
    irreps={
        "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "A2g": [1, 1,-1, -1, 1,	1, -1, 1, 1, -1],
        "Eg":  [2, -1, 0, 0, 2, 2, 0, -1, 2, 0],
        "T1g": [3, 0, -1, 1, -1, 3, 1, 0, -1, -1],
        "T2g": [3, 0, 1, -1, -1, 3, -1, 0, -1, 1],

        "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
        "A2u": [1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
        "Eu":  [2, -1, 0, 0, 2, -2, 0, 1, -2, 0],
        "T1u": [3, 0, -1, 1, -1, -3, -1, 0, 1, 1],
        "T2u": [3, 0, 1, -1, -1, -3, 1, 0, 1, -1],
    }
)


Td = PointGroup(
    name="Td",
    classes=["E", "8C3", "3C2", "6S4", "6σd"],
    n_ops=[1, 8, 3, 6, 6],
    irreps={
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "E":  [2, -1, 2, 0, 0],
        "T1": [3, 0, -1, 1, -1],
        "T2": [3, 0, -1, -1, 1],
    }
)

# -------------------------
#CLI 
# -------------------------
gamma_red = input("Enter the reducible representation (comma-separated values): ")
gamma_red = [int(x) for x in gamma_red.split(",")]

group = input("Enter the point group: ")
if group == "C4v":
    decomposition = C4v.reduce_representation(gamma_red)
elif group == "C3v":
    decomposition = C3v.reduce_representation(gamma_red)
elif group == "C2v":
    decomposition = C2v.reduce_representation(gamma_red)
elif group == "Oh":
    decomposition = Oh.reduce_representation(gamma_red)
elif group == "Td":
    decomposition = Td.reduce_representation(gamma_red)


print("Decomposition into irreducible representations:")
for irrep, coeff in decomposition.items():
    if coeff != 0:
        print(f"{coeff} × {irrep}")