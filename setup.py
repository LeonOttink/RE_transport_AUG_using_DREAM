# Author: Leon Ottink (TUe), Mathias Hoppe (KTH)
#
# Object for setting up DREAM settings for ASDEX Upgrade disruption, using 
# input parameters from command line from parser, as well as userInput.py.
###############################################################################


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.constants import mu_0, c, e, pi
import sys

sys.path.append('../py/') # go to local DREAM module location

from DREAM import DREAMSettings, runiface, runiface_parallel, DREAMOutput
import DREAM.Settings.AdvectionInterpolation as AI
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.OhmicCurrent as OhmicCurrent
import DREAM.Settings.Equations.RunawayElectrons as RunawayElectrons
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TransportSettings as Transport

import userInput as ui


class SetUp():


    def __init__(self, args):
        """
        Create object containing DREAM settings, with methods for 
        applying settings appropriate for ASDEX Upgrade disruptions.

        Parameters
        ----------
        args : argparse namespace
            Namespace containing all tunable parameters to be applied 
            here in the DREAM settings object.
        """
        # Retrieve physical parameters
        self.a       = ui.a          # minor radius (m)
        self.b       = args.b        # effective wall radius (m)
        self.B0      = ui.B0         # on-axis magnetic field (T)
        self.Ip      = ui.Ip         # total plasma current (A)
        self.nAr     = args.nAr      # argon impurity density (m^-3)
        self.R0      = ui.R0         # major radius (m)
        self.sigma   = args.sigma    # impurity density peakedness (#)
        self.tauwall = args.tauwall  # wall resistive time (s)
        self.Vloop   = ui.Vloop      # externally applied loop voltage (V)
        
        self.impurities = ui.impurities
        self.j0r, self.j0 = ui.getCurrentDensity(r=None, nr=100)
        self.n0r, self.n0 = ui.getInitialDensity(r=None, nr=100)
        self.T0r, self.T0 = ui.getInitialTemperature(r=None, nr=100)
        
        ## Retrieve magnetic perturbation parameters
        self.A0    = args.A0    # initial amplitude of dBB(t) (#)
        self.A1    = args.A1    # final amplitude of dBB(t) (#)
        self.dBB   = args.dBB   # magnetic perturbation array or callable (#)
        self.dBB_r = args.dBB_r # array of radii corresponding to dBB (m)
        self.dBB_t = args.dBB_t # array of times corresponding to dBB (s)
        self.idBBr = args.idBBr # index of radial profile as specified in userInput (#)
        self.tco   = args.tco   # cut-off time in dBB(t) which is a high-low step function (s)
        self.Dre   = None       # RE diffusion coefficient (m^2/s)
        self.Are   = None       # RE advection coefficient (m/s)
        
        # Retrieve numerical parameters
        self.dt0    = args.dt0     # (initial) time step (s)
        self.dtMax  = args.dtMax   # maximum time step in addaptive stepper (s)
        self.Np     = args.Np      # number of momentum space grid points (#)
        self.Nr     = args.Nr      # number of radial grid points (#)
        self.pMax   = args.pMax    # cut-off value momentum space (mc)
        self.reltol = args.reltol  # relative tollerance of DREAM solvers (#)
        self.tMax   = args.tMax    # simulation time (s)
        
        # Retrieve user settings
        self.extension   = args.extension   # extension to DREAM output and settings paths
        self.prefix      = args.prefix      # prefix to DREAM output and settings paths
        self.quiet       = args.quiet       # wheter to completely silence DREAM outputs during simulation
        self.timestepper = args.timestepper # wether to use constant or adaptive stepper
        self.verbose     = args.verbose     # wether to show extended DREAM outputs during simulation
        self.visualize   = args.visualize   # wheter to visualize intermediate steps
        
        # Initialize DREAM settings
        self.ds = DREAMSettings()
    
    def apply_settings(self, verbose=False, viz=False):
        """
        Generate DREAM simulation settings for a disruption.
        
        Parameters
        ----------
        verbose : bool, optional, defaults to False
            Wheter to print additional DREAM metadata during simulation.
        viz : bool, optional, defaults to False
            Wheter to visualize some intermediate results during initial
            phase of the simulation.
        
        Returns
        -------
        ds : DREAM settings object
            Contains all the initializations, boundary conditions and 
            settings for the disruption simulation.
        """
        ## Initialization
        # Collision settings, mostly influence calculation of E_{c,eff}
        # bremsstrahlung losses and partial ionization effects are taken into 
        # account (most general case as implemented in DREAM). Energy dependent
        # model for Coulomb logarithm is used to accomodate for the broad range
        # of electron energies.
        self.ds.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
        self.ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED 
        self.ds.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT
        # p* is the critical momentum. Collisionless mode keeps the effect of 
        # trapped particle orbits canceling the electric field and thus not 
        # becoming RE, collisional removes this effect. Both are 
        # approximations, not necessarily one better.
        self.ds.collisions.pstar_mode = Collisions.PSTAR_MODE_COLLISIONLESS

        # Setup grid and initialize E-field and j-profile/fhot in short sims
        self.setup_toroidalgrid(viz=viz)
        INITFILE = self.setup_initial(viz=viz)
        # Use super-thermal limit (Tcold-->0) for collision frequencies
        self.ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_SUPERTHERMAL
        
        # Initialize settings for TQ sim with output from init sim
        # Some quantities must not be initialized this way.
        # (different kinds of sims not compatible for all parameters)
        ignorelist = ['n_i', 'N_i', 'T_i', 'W_i', 'n_cold', 
                      'T_cold', 'W_cold', 'n_hot']
        self.ds.fromOutput(INITFILE, ignore=ignorelist)
        
        ## Physics parameter settings
        # General models for Eceff and avalanche
        self.ds.eqsys.n_re.setEceff(RunawayElectrons.COLLQTY_ECEFF_MODE_FULL) # most general
        self.ds.eqsys.n_re.setAvalanche(RunawayElectrons.AVALANCHE_MODE_FLUID_HESSLOW)
        # Dreicer generation is already included in kinetics!
        self.ds.eqsys.n_re.setDreicer(RunawayElectrons.DREICER_RATE_DISABLED)
        # Initialize T_cold to arbitrarily low value
        self.ds.eqsys.T_cold.setInitialProfile(temperature=1)
        # Let all electrons contribute to ionization, but use simpler jaccobian
        # reducing high computational cost against little difference
        self.ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_KINETIC_APPROX_JAC)
        # Set self-consistent E and T evolution
        self.ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
        self.ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
        self.ds.eqsys.j_ohm.setCurrentProfile(self.j0, radius=self.j0r, 
                                              Ip0=self.Ip)
        
        t, r, dBB, self.Dre, self.Are = self.getDiffusionAdvectionRE(
            dBB=self.dBB, t=self.dBB_t, r=self.dBB_r)
        self.ds.eqsys.T_cold.transport.setMagneticPerturbation(dBB=dBB, t=t, r=r)
        self.ds.eqsys.f_hot.transport.setMagneticPerturbation(dBB=dBB, t=t, r=r)
        self.ds.eqsys.n_re.transport.prescribeDiffusion(drr=self.Dre, t=t, r=r)
        self.ds.eqsys.n_re.transport.prescribeAdvection(ar=self.Are, t=t, r=r)
        
        # Add arbitrarily cold injected impurities
        for im in self.impurities:
            self.ds.eqsys.n_i.addIon(
                im['name'], Z=im['Z'], iontype=Ions.IONS_DYNAMIC_NEUTRAL, 
                n=im['n'](im['r'],self.nAr,self.sigma), r=im['r'], T=1)
        
        # Set transport boundary conditions
        self.ds.eqsys.T_cold.transport.setBoundaryCondition(bc=Transport.BC_F_0)
        self.ds.eqsys.f_hot.transport.setBoundaryCondition(bc=Transport.BC_F_0)
        self.ds.eqsys.n_re.transport.setBoundaryCondition(bc=Transport.BC_F_0)
        
        # Set boundary conditions to the E-field
        if self.tauwall is None:
            Vloop_wall_R0 = 0
            if self.Vloop is not None:
                Vloop_wall_R0 = self.Vloop / self.R0

            self.ds.eqsys.E_field.setBoundaryCondition(
                ElectricField.BC_TYPE_PRESCRIBED,
                V_loop_wall_R0=Vloop_wall_R0)
        elif self.Vloop is None or self.Vloop == 0:
            ds.eqsys.E_field.setBoundaryCondition(
                ElectricField.BC_TYPE_SELFCONSISTENT,
                inverse_wall_time=1/self.tauwall, R0=self.R0)
        else:
            ds.eqsys.E_field.setBoundaryCondition(
                ElectricField.BC_TYPE_TRANSFORMER,
                inverse_wall_time=1/self.tauwall, R0=self.R0,
                V_loop_wall_R0=self.Vloop/self.R0)

        ## Solver settings
        # Setup solver and tolerances
        self.ds.solver.setType(Solver.NONLINEAR)
        self.ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_MKL)
        self.ds.solver.setMaxIterations(500)
        
        MAXIMUM_IGNORABLE_ELECTRON_DENSITY = 1e5
        self.ds.solver.tolerance.set(reltol=self.reltol)
        self.ds.solver.tolerance.set(unknown='n_re', reltol=self.reltol, 
            abstol=MAXIMUM_IGNORABLE_ELECTRON_DENSITY)
        self.ds.solver.tolerance.set(unknown='j_re', reltol=self.reltol, 
            abstol=e*c*MAXIMUM_IGNORABLE_ELECTRON_DENSITY)
        self.ds.solver.tolerance.set(unknown='f_re', reltol=self.reltol, 
            abstol=MAXIMUM_IGNORABLE_ELECTRON_DENSITY)
        self.ds.solver.tolerance.set(unknown='f_hot',reltol=self.reltol, 
            abstol=MAXIMUM_IGNORABLE_ELECTRON_DENSITY)
        self.ds.solver.tolerance.set(unknown='n_hot',reltol=self.reltol, 
            abstol=MAXIMUM_IGNORABLE_ELECTRON_DENSITY)
        self.ds.solver.tolerance.set(unknown='j_hot',reltol=self.reltol, 
            abstol=e*c*MAXIMUM_IGNORABLE_ELECTRON_DENSITY)

        # Print and store timing info
        self.ds.output.setTiming(True, True)
        self.ds.solver.setVerbose(verbose)
        
        # Setup time stepper
        if self.timestepper == 'constant':
            # Reset Nt to None after initial sims so that dt can be specified
            self.ds.timestep.setNt(None)
            self.ds.timestep.setDt(self.dt0)
            self.ds.timestep.setTmax(self.tMax)
        else: # addaptive time stepper
            self.ds.timestep.setIonization(dt0=self.dt0, dtmax=self.dtMax, 
                                           tmax=self.tMax)
        
        ## Save settings and generate output file name
        self.ds.save(self.outname(phase='', prefix=self.prefix, io='settings', 
                                  extension=self.extension))
        OUTFILE = self.outname(phase='', prefix=self.prefix, io='output', 
                               extension=self.extension)

        return self.ds, OUTFILE
        
    def setup_toroidalgrid(self, rGridNonuniformity=1, nr=100, viz=False):
        """
        Set the radial grid of the given DREAM settings object to 
        correspond to a typical magnetic field as provided in 
        userInput.py.
        
        Parameters
        ----------
        rGridNonuniformity : scalar, optional, defaults to 1
            Determines how uniform toroidal flux surfaces are 
            distributed in the poloidal plane.
        nr : int, optional, defaults to 100
            Radial grid resolution for analytical magnetic field.
        viz : boolean, optional, defaults to False
            Show a plot of the computed magnetic field if True.
        """

        # Radial grid for analytical magnetic field
        r = np.linspace(0, self.a, nr)

        # Elongation profile
        kappa = ui.getKappa(nr)

        # Poloidal flux radial grid
        psi_p = -mu_0 * self.Ip * (1-(r/self.a)**2) * self.a

        self.ds.radialgrid.setType(RadialGrid.TYPE_ANALYTIC_TOROIDAL)
        self.ds.radialgrid.setWallRadius(self.b)
        self.ds.radialgrid.setMajorRadius(self.R0)

        # Minor radius is set as bound in explicitly set grid points
        r_f = np.linspace(0, self.a**rGridNonuniformity, self.Nr+1)
        r_f = r_f**(1/rGridNonuniformity) # optional sqeeze/stretch of grid
        r_f = r_f * self.a/r_f[-1] # correct for roundoff
        self.ds.radialgrid.setCustomGridPoints(r_f)

        self.ds.radialgrid.setShaping(psi=psi_p, rpsi=r, GOverR0=self.B0, 
                                      kappa=kappa, rkappa=r)

        if viz:
            self.ds.radialgrid.visualize(ntheta=200)
    
    def setup_initial(self, verbose=False, viz=False):
        """
        Initializes pre-disruption electric field and current density 
        profile.
        
        Parameters
        ----------
        verbose : boolean, optional, defaults to False
            Print out DREAM meta data during simulation.
        viz : boolean, optional, defaults to False
            Plot intermediate results. 
            
        Returns
        -------
        INITFILE : str
            Path to output file of initialization simulation.
        """
        ## Compute Efield consistent with a prescribed current profile
        # A 1-step simulation computes the conductivity required for 
        # calculating Efield
        print('Electric field initialization...')
        self.ds.eqsys.E_field.setType(ElectricField.TYPE_PRESCRIBED_OHMIC_CURRENT)
        self.ds.eqsys.j_ohm.setCurrentProfile(self.j0, radius=self.j0r, Ip0=self.Ip)
        self.ds.eqsys.j_ohm.setConductivityMode(OhmicCurrent.CONDUCTIVITY_MODE_SAUTER_COLLISIONAL)
        
        # Disable kinetic grids during conductivity simulation as not needed
        self.ds.runawaygrid.setEnabled(False)
        self.ds.hottailgrid.setEnabled(False)
        
        # The initial temperature profile only enters in the shape of the 
        # initial Maxwellian on the hot grid. This temperature is that of the 
        # cold population in the TQ simulation.
        self.ds.eqsys.T_cold.setPrescribedData(temperature=self.T0, 
                                               radius=self.T0r)
        
        # Set main ion density
        self.ds.eqsys.n_i.addIon(name='D', Z=1, Z0=1, 
            iontype=Ions.IONS_DYNAMIC, T=self.T0, r=self.T0r, n=self.n0)
        
        # Setup and run simulation to find electric field
        self.ds.timestep.setTmax(1e-11)
        self.ds.timestep.setNt(1)
        self.ds.other.include('fluid', 'scalar')
        do = runiface(self.ds, quiet=not verbose)
        
        # Store electric field profile
        Einit = do.eqsys.E_field[-1,:]
        Einit_r = do.grid.r[:]
        do.close()
        
        ## Compute fhot equilibrium consistent with prescribed Efield
        print('Pre-disruption equilibrium initialization...')
        # Prescribe electric field profile
        # (set E here instead of j to allow f_hot to be properly initialized)
        self.ds.eqsys.E_field.setPrescribedData(Einit, radius=Einit_r)
        
        # Setup hottailgrid for current simulation
        self.ds.hottailgrid.setEnabled(True)
        self.ds.hottailgrid.setNxi(1) # isotropic mode
        self.ds.hottailgrid.setPmax(self.pMax)
        self.ds.hottailgrid.setNp(self.Np)
        self.ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
        self.ds.eqsys.j_ohm.setCorrectedConductivity(False)
        
        # Initialize hot electron distribution conserving quasi-neutrality
        nfree, rn0 = self.ds.eqsys.n_i.getFreeElectronDensity()
        self.ds.eqsys.f_hot.setInitialProfiles(rn0=rn0, 
            n0=(1-1e-3)*nfree*np.ones(rn0.shape), T0=self.T0, rT0=self.T0r)
        self.ds.eqsys.f_hot.setBoundaryCondition(bc=FHot.BC_F_0)
        self.ds.eqsys.f_hot.setAdvectionInterpolationMethod(
            ad_int=AI.AD_INTERP_TCDF, ad_jac=AI.AD_INTERP_JACOBIAN_UPWIND)
        self.ds.eqsys.n_re.setAdvectionInterpolationMethod(
            ad_int=AI.AD_INTERP_TCDF, ad_jac=AI.AD_INTERP_JACOBIAN_UPWIND)
        # Exclude Jacobian elements df_hot/dn_i to spare lot of space in matrix
        self.ds.eqsys.f_hot.enableIonJacobian(False)

        # Save settings and setup and run simulation to find electric field
        self.ds.timestep.setTmax(1) # Steady state ohmic current (let t -> inf)
        self.ds.timestep.setNt(3)
        self.ds.solver.setVerbose(verbose)
        self.ds.save(self.outname(phase='_init', prefix=self.prefix, 
                                  io='settings', extension=self.extension))
        INITFILE = self.outname(phase='_init', prefix=self.prefix, io='output', 
                                extension=self.extension)
        do = runiface(self.ds, INITFILE, quiet=not verbose)    
        do.close()
        
        return INITFILE
        
    def getDiffusionAdvectionRE(self, dBB=0, t=None, r=None, nt=100, nr=99, TaylorOrder=7):
        """
        Assume safety profile q=1 everywhere in the plasma, and that 
        all RE move parallel to the magnetic field with lightspeed v=c.
        Assume circular plasma for advection.
        
        Parameters:
        -----------
        dBB : scalar, np.ndarray or callable, optional, defaults to 0
            Magnetic turbulance amplitude. If not constant, 
            its shape should match at least one of t or r.
        t : scalar or np.ndarray, optional, defaults to None
            Time (s).
        r : scalar or np.ndarray, optional, defaults to None
            Minor radius (m).
        nr, nt : int, optional, defaults to 100
            Radial and time grid resolution to use if r and/or t are/is 
            None.
        TaylorOrder : int, optional, defaults to 7
            Order of Taylor expansion used to evaluate A. Max 7.
            
        Returns
        -------
        t : None, scalar or np.ndarray
            Time points at which D and A are evaluated. If None D 
            and A are not time-dependent.
        r : scalar or np.ndarray
            Radial grid points at which D and A are evaluated.
        dBB : np.ndarray of shape [len(t),len(r)]
            Magnetic perturbation array.
        D : scalar or np.ndarray of shape [len(t),len(r)]
            RE diffusion coefficients at radii r.
        A : scalar or np.ndarray of shape [len(t),len(r)]
            RE advection coefficients at radii r.
        """
        t, r, dBB = self.getArrays(coeff=dBB, t=t, r=r, nr=nr, nt=nt)
        
        D = pi*self.R0*c*dBB**2

        # 7th order Taylor expansion in epsilon=r/R0 or <B/Bmin> results in:
        if TaylorOrder > 7: 
            TaylorOrder = 7; print("WARNING: Max TaylorOrder = 7")
        c1 = np.array([1,1,1.5,1.5,3.75 ,3.75 ,2.1875 ,2.1875 ])[:TaylorOrder+1]
        c2 = np.array([1,1,0.5,0.5,0.375,0.375,0.15625,0.15625])[:TaylorOrder+1]
        
        A = -D/self.R0*np.polyval(np.flip(c1), r/self.R0) \
            /np.polyval(np.flip(c2), r/self.R0)
        
        return t, r, dBB, D, A

    def getArrays(self, coeff, t=None, r=None, nt=80, nr=100):
        """
        Ensures that the given input coefficient(s) are of valid 
        dimensions considering their time and radial position 
        dependence.
        """
        isarr = lambda x: isinstance(x,np.ndarray)
        iscon = lambda x: np.isscalar(x)

        if t is None:
            t = np.linspace(0, self.tMax, nt)
        if r is None:
            r = np.linspace(0, self.a, nr)

        if callable(coeff): 
            coeff = coeff(self.A0, self.tco, self.A1, self.idBBr)

        if iscon(t) and iscon(r):
            # r given as constant and t constant --> output scalar
            if not iscon(coeff): 
                print("Warning: r and t are assumed scalar or None but coeff "
                      "is no scalar!")
        elif isarr(t) and iscon(r):
            # t given as array or None and r as constant requires coeff to be 
            # compatible with array shape of t
            if iscon(coeff): 
                coeff *= np.ones([t.size,1])
            elif coeff is not None and coeff.size == t.size: 
                coeff = coeff.reshape([t.size,1])
            else: print("Warning: coeff and t should match in size if r "
                        "constant!")
        elif iscon(t) and isarr(r):
            # r given as array or None and thereby constructed as array, and t 
            # constant, requires coeff to be compatible with array shape of r
            if iscon(coeff): 
                coeff *= np.ones([1,r.size])
            elif coeff is not None and coeff.size == r.size: 
                coeff = coeff.reshape([1,r.size])
            else: print("Warning: coeff and r should match in size if t "
                        "constant or None!")
        elif isarr(t) and isarr(r):
            # r and t given as arrays (or constructed as such) requires coeff 
            # to be compatible with 2D (t,r)-shape
            if iscon(coeff): coeff *= np.ones([t.size,r.size])
            # Assume that if coeff same shape as both t and r, it is a function 
            # of t, not r
            elif coeff is not None and coeff.shape == t.shape: 
                coeff = np.stack([coeff]*r.size).T
            elif coeff is not None and coeff.shape == r.shape: 
                coeff = np.stack([coeff]*t.size)
            elif coeff is not None and coeff.shape != (t.size,r.size): 
                print("Warning: the shape of coeff does not match that of r "
                      "and/or t!")

        return t, r, coeff
        
    @staticmethod
    def outname(prefix, phase, io='', extension=''):
        """
        Generate an output file name.
        """
        filename = f'{prefix}{io}{phase}{extension}.h5'
        p = Path(filename).parent.resolve()

        if not p.exists():
            p.mkdir(parents=True)

        return filename

