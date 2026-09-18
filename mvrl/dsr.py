"""DSR signal and finite-horizon validity; legacy equivalence claim withdrawn."""
import numpy as np

def dsr_step(A,B,rho,eta=.05):
    """Moody–Saffell differential signal (without constant eta multiplier)."""
    if not 0<eta<1:raise ValueError('eta must lie in (0,1)')
    V=np.asarray(B)-np.asarray(A)**2
    if np.any(V<=0):raise ValueError('strictly positive initial variance required')
    D=(B*(rho-A)-.5*A*(rho*rho-B))/V**1.5
    return D,A+eta*(rho-A),B+eta*(rho*rho-B)

def ewma_variance(variance,eta,steps):
    return variance*eta/(2-eta)*(1-(1-eta)**(2*steps))

if __name__=='__main__':
    print('DSR is not dynamically equivalent to a fixed quadratic penalty.')
    print('Run python -m mvrl.experiments for exact bounded-tree comparisons.')
