# this code is based on MATLAB implementation that uses analytic solutions (via elliptic integrals) to calculate the magnetic field of a solenoid

import numpy as np
import matplotlib.pyplot as plt
from scipy import special




def magnetic_field_loop(r, z, a):
    """
    Calculates magnetic field due to a single current loop.

    Args:
        r: Radial distance from the loop's axis.
        z: Axial distance from the loop's center.
        a: Radius of the loop.

    Returns:
        Tuple of magnetic field components (Br, Bz).

    Based on analytic solution of the Biot-Savart Law for a current loop
    (makes use of the elliptical integral function so requires scipy)
    For the maths, see: https://patents.google.com/patent/EP0310212A2/en
                             or
    https://www.grant-trebbin.com/2012/04/off-axis-magnetic-field-of-circular.html
    
    Daniel Gallichan, July 2024

    """

    thisR = r.copy()  # we don't want to modify our input!

    thisR[thisR < 0] = -thisR[thisR < 0]  # Invert r for negative values

    term1 = 1 / np.sqrt((a+thisR)**2 + z**2)
    term2 = 1 / ((a-thisR)**2 + z**2)

    m = 4 * a * thisR / ((a + thisR)**2 + z**2)

    m = np.abs(m)
    m[m > 1] = 1 # can become larger than 1 due to numerical errors

    E1 = special.ellipk(m)
    E2 = special.ellipe(m)

    Bz = (1/np.pi) * term1 * ( E2*(a**2 - thisR**2 - z**2)*term2 + E1 )
    Br = (z/thisR) * (1/np.pi) * term1 * ( E2*(a**2 + thisR**2 + z**2)*term2 - E1 )

    Br[thisR < 1e-6] = 0  
    Br[r<0] = -Br[r<0]  # Invert Br for negative r values

    return Br, Bz

def magnetic_field_solenoid(r, z, N, I, L, a):
    """
    Approximates magnetic field of a solenoid using multiple loops.

    Args:
        r: Radial distance from the solenoid axis.
        z: Axial distance from the solenoid center.
        N: Number of turns.
        I: Current.
        L: Length of solenoid.
        a: Radius of each loop.

    Returns:
        Tuple of magnetic field components (Br, Bz).
    """

    dz = L / N  # Spacing between loops
    z_loops = np.linspace(-L/2, L/2, N)
    Br, Bz = 0, 0
    for z_loop in z_loops:
        br, bz = I * magnetic_field_loop(r, z - z_loop, a)
        Br += br
        Bz += bz
    return Br, Bz



N = 100  # Number of loops in magnet
I = 1   # Current
L = .05   # Length of solenoid (in x direction)
a = .015 # magnet radius

# Create a grid of points
x_max = .15
y_max = .15
x = np.linspace(-x_max, x_max, 150)
y = np.linspace(-y_max, y_max, 150)
X, Y = np.meshgrid(x, y)


# Calculate magnetic field components
Br, Bx = magnetic_field_solenoid(Y, X, N, I, L, a)
# Br, Bx = magnetic_field_loop(Y, X, a)


fieldMag = np.sqrt(Bx**2 + Br**2)
magThresh = np.percentile(fieldMag, 95)

# Plot the magnetic field vectors
p1 = plt.subplot2grid((2, 2), (0, 0))
p2 = plt.subplot2grid((2, 2), (1, 0))
p3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

climMax = .3*magThresh
clims = (-climMax, climMax)
im1 = p1.imshow(Bx,extent=[-x_max, x_max, -y_max, y_max],origin='lower',clim=clims)
p1.set_aspect('equal')
p1.set_xlabel('x')
p1.set_ylabel('y')
p1.set_title('Bx')
plt.colorbar(im1, ax=p1)

im2 = p2.imshow(Br,extent=[-x_max, x_max, -y_max, y_max],origin='lower',clim=clims)
p2.set_title('By')
plt.colorbar(im2, ax=p2)

# to make the quiver plot, need to rescale the vectors
# to make them visible

# rescale the vectors by an arbitrary power:
powFact = .3
BxScaled = (Bx / fieldMag) * (fieldMag / magThresh)**powFact
BrScaled = (Br / fieldMag) * (fieldMag / magThresh)**powFact
BxScaled[fieldMag > magThresh] = 0
BrScaled[fieldMag > magThresh] = 0
skip = 3 # only plot a subset of the vectors on the quiver
p3.streamplot(X, Y, Bx, Br,broken_streamlines=False,zorder=1)
p3.quiver(X[::skip,::skip], Y[::skip,::skip], BxScaled[::skip,::skip], BrScaled[::skip,::skip],pivot='mid',scale=20,zorder=2)
p3.set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()





