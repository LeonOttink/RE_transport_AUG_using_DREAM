import numpy as np

###########################################################################
# TODO: Change these for your simulation
param = ''     # Match one of the command line parameters as set in disrupt
unit  = ''     # Unit of parameter for headers
scan  = ''     # Scan number for file naming
scan_list = [] # Scan list of values for param to loop over
###########################################################################

## Plasma parameters
a  = 0.5        # minor radius (m)
b  = 2.5        # (effective) wall radius (m)
B0 = 2.5        # on-axis magnetic field strength (T)
R0 = 1.65       # major radius (m)
kappa0 = 1.0    # elongation on axis (#)
kappaa = 1.15   # elongation at edge (#)

Ip = 800e3      # target plasma current (A)
j0 = 1.52e6     # initial current density (A/m^2)
tauwall = None  # wall resistive decay time (s)
Vloop   = None  # loop voltage (V)

# Ti0 = Te0 and ni0 = ne0 are assumed, since no seperate electron or ion
# heating takes place and a pure and fully ionized D plasma is assumed
ne0 = 2.6e19    # central electron density (m^-3)
Te0 = 5.8e3     # central electron temperature (eV)
Drr = 4e3       # heat diffusion coefficient (m^2/s)

# Normalized exponential impurity density profile with sigma the edge-peakedness (0 gives flat profile)
norm = lambda sigma: (2*(sigma+np.exp(-sigma)-1)/sigma**2) if abs(sigma)>1e-7 else 1
n_im = lambda r, nAr0, sigma: nAr0*np.exp(-sigma*(1-r/a))/norm(sigma)
impurities = [{'name': 'Ar',   'Z': 18, 'n': n_im, 'r': np.linspace(0,a,100)}] # added injected impurities

## Numerical parameters
pMax  = 1.0     # maximum momentum (m_e*c)
Np    = 80      # number of momentum grid points (#)
Nxi   = 1       # number of pitch grid points (#)
Nr    = 15      # number of radial grid points in simulation (#)
tMax  = 5e-3    # simulation time (s)
dt0   = 2e-11   # starting time step (s)
dtMax = 1e-6    # maximum time step (s)
reltol = 1e-6   # solver tolerance

## Profiles
# Magnetic perturbation profile
dBB_r = np.linspace(0,a,15)
# Sim has microsecond precision, no tco<1e-6!
dBB_t = np.concatenate([np.logspace(np.log10(dt0),np.log10(1e-6),51),
                        np.linspace(2e-6,99e-6,98),
                        np.linspace(1e-4,9e-4,9),
                        np.logspace(np.log10(1e-3),np.log10(tMax),42)])

dBBr = {0: np.ones(dBB_r.size),                         # 0: constant
        1: 0.1+np.exp(-((dBB_r-0.33)/0.09)**2/2)  # 1: islands at q=2/1 and 3/2 with half-widths matching (Boozer
          +np.exp(-((dBB_r-0.43)/0.13)**2/2)}
# Normalize profiles
for k in dBBr.keys():
    dBBr[k] *= dBBr[k].size/dBBr[k].sum()

dBB = lambda A0=0,tco=0.,A1=0,i=0: np.outer((dBB_t<=tco)*A0 + (dBB_t>tco)*A1, dBBr[i])

# ASDEX-U elongation profile
getKappa = lambda nr: np.linspace(kappa0, kappaa, nr)

def getInitialTemperature(r=None, nr=100):
    """
    Returns the initial ASDEX Upgrade temperature profile.
    
    Parameters
    ----------
    r : scalar or numpy.ndarray, optional, defaults to None
        Minor radius (m).
    nr : int, optional, defaults to 100
        Radial grid resolution to use if ``r = None``.
        
    Returns
    -------
    r : scalar or np.ndarray
        Radial grid points at which temperature is evaluated.
    Te0 : scalar or np.ndarray
        Initial temperature profile at radii r.
    """
    if r is None:
        r = np.linspace(0, a, nr)
        
    c1 = 0.4653 # T(a)=57 eV
#    c1 = 0.505 # T(a)=115 eV
    return r, Te0 * np.exp(-((r/a)/c1)**2)

def getInitialDensity(r=None, nr=100):
    """
    Returns the initial ASDEX Upgrade density profile.
    
    Parameters
    ----------
    r : scalar or numpy.ndarray, optional, defaults to None
        Minor radius (m).
    nr : int, optional, defaults to 100
        Radial grid resolution to use if ``r = None``.
        
    Returns
    -------
    r : scalar or np.ndarray
        Radial grid points at which density is evaluated.
    ne0 : scalar or np.ndarray
        Initial density profile at radii r.
    """
    if r is None:
        r = np.linspace(0, a, nr)

    return r, ne0 * np.polyval([-0.1542, -0.01115, 1.], r/a)

def getCurrentDensity(r=None, nr=100, p=4, q=1.5):
    """
    Returns the ASDEX Upgrade current density profile.
    
    Parameters
    ----------
    r : scalar or numpy.ndarray, optional, defaults to None
        Minor radius (m).
    nr : int, optional, defaults to 100
        Radial grid resolution to use if ``r = None``.
        
    Returns
    -------
    r : scalar or np.ndarray
        Radial grid points at which current density is evaluated.
    j0 : scalar or np.ndarray
        Current density profile at radii r.
    """
    if r is None:
        r = np.linspace(0, a, nr)
    
    # ASDEX-U undesturbed profile: p=4, q=1.5
    return r, j0 * (1 - (r/a)**p)**q
