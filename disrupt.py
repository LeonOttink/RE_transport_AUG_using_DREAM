# Author: Leon Ottink (TUe), Mathias Hoppe (KTH)
#
# Main script for running isotropic DREAM simulations of ASDEX Upgrade
# disruptions including magnetic turbulence, with the goal to estimate 
# the radial transport of runaway electrons in these events.
###############################################################################


import argparse
from copy import deepcopy
import os
from setup import *
from warnings import warn

def parse_args(argv=None):
    """
    Terminal input argument parser.

    Parameters
    ----------
    argv : list of command line arguments
    
    Returns
    -------
    args : object storing parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Run a disruption simulation")
        
    # Output arguments
    parser.add_argument('-v', '-verbose', help="Wether to make simulations verbose", dest="verbose", type=bool)
    parser.add_argument('-q', '-quiet', help="Make DREAM completely quiet", dest="quiet", type=bool)
    parser.add_argument('-viz', '-visualize', help="Wether to show intermediate plots", dest="visualize", type=bool)
    parser.add_argument('-pre','-prefix', help="Prefix to settings and output file name", dest="prefix", action="store", type=str)
    parser.add_argument('-ext','-extension', help="Extension to settings and output file name", dest="extension", action="store", type=str)
    
    # Perturbation profile arguments
    parser.add_argument('-dbb','-dBB', help="Magnetic turbulence amplitude", dest="dBB", action="store", type=float)
    parser.add_argument('-dbb_t','-dBB_t', help="Times corresponding to magnetic turbulence", dest="dBB_t", action="store", type=float)
    parser.add_argument('-dbb_r','-dBB_r', help="Radii corresponding to magnetic turbulence", dest="dBB_r", action="store", type=float)
    parser.add_argument('-tco','-tcutoff', help="Cutoff time of dBB profile", dest="tco", action="store", type=float)
    parser.add_argument('-A0','-dBB0', help="Initial dBB before cutoff time", dest="A0", action="store", type=float)
    parser.add_argument('-A1','-dBB1', help="Final dBB after cutoff time", dest="A1", action="store", type=float)
    parser.add_argument('-ir','-idBBr', help="Index of dBB radial profile to use as specified in user input", dest="idBBr", action="store", type=int)
    
    # Bisection algorithm arguments
    parser.add_argument('-bisect','-bisection', help="Boundaries a and b for bisection algorithm", dest="bisect", action="store", nargs=2, type=float)
    parser.add_argument('-max_iter','-max_iterations', help="Maximum number of iterations in bisection algorithm", dest="max_iter", action="store", type=int)
    
    # Physical parameters
    parser.add_argument('-b', help="Wall radius", dest="b", action="store", type=float)
    parser.add_argument('-tauwall', help="Wall resistive time", dest="tauwall", action="store", type=float)
    parser.add_argument('-nAr', help="Injected Argon density", dest="nAr", action="store", type=float)
    parser.add_argument('-sig','-sigma', help="Peakedness of impurity density profile", dest="sigma", action="store", type=float)

    # Numerical parameters
    parser.add_argument('-pmax','-pMax', help="Momentum space cut-off", dest="pMax", action="store", type=float)
    parser.add_argument('-reltol', help="Relative tolerance of DREAM solvers", dest="reltol", action="store", type=float)
    parser.add_argument('-np','-Np', help="Number of momentum grid points", dest="Np", action="store", type=int)
    parser.add_argument('-nxi','-Nxi', help="Number of pitch grid points", dest="Nxi", action="store", type=int)
    parser.add_argument('-nr','-Nr', help="Number of radial grid points", dest="Nr", action="store", type=int)
    parser.add_argument('-dt0', help="Initial time step", dest="dt0", action="store", type=float)
    parser.add_argument('-dtMax', help="Highest allowed time step", dest="dtMax", action="store", type=float)
    parser.add_argument('-tMax', help="Simulation time", dest="tMax", action="store", type=float)
    parser.add_argument('-timestepper', help="Wether to use 'constant' time step or 'adaptive'", dest='timestepper', action='store', type=str)
    parser.add_argument('-nthreads','-Nthreads', help="Number of threads to limit sims to", dest='Nthreads', action='store', type=int)
    parser.add_argument('-njobs','-Njobs', help="Number of jobs to use for sims", dest='Njobs', action='store', type=int)
    parser.add_argument('-sscan','-sensitivityScan', help="Wether to scan around the given A0,p2 settings for sensitivy measurements", dest='sensitivity_scan', action='store', type=bool)
    
    # Setup defaults
    parser.set_defaults(
        verbose=False, quiet=False, visualize=False, sensitivity_scan=False,
        prefix='output/', extension='', Nthreads=None, Njobs=6,
        dBB=ui.dBB, dBB_t=ui.dBB_t, dBB_r=ui.dBB_r, 
        tco=0., A0=0., idBBr=0, A1=0., bisect=[None,None], max_iter=20,
        b=ui.b, tauwall=ui.tauwall, nAr=8.3e19, sigma=0.,
        pMax=ui.pMax, reltol=ui.reltol, Np=ui.Np, Nxi=ui.Nxi, Nr=ui.Nr,
        dt0=ui.dt0, dtMax=ui.dtMax, tMax=ui.tMax, timestepper='adaptive')

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        warn(f'Argument parser received unknown arguments {unknown} which will not be used!')
    return args
    

def bisectionParallel(f, A, B, P2, tol=50, max_iter=1, filename='bisectionData.txt'):
    """
    Run bisection algorithm on function `fun` in parallel for different 
    p2 given as list, with possibly different starting a and b given as 
    list of equal length.
    
    Parameters
    ----------
    f : callable 
        f should have inputs A0,P2: lists of equal length, A0 the 
        initial dB/B amplitude and P2 a second parameter, and output 
        errors: a list of floats used as error measure in the bisection 
        algorithm.
    A, B : lists of a and b values (floats) of equal length
        [a,b] is the initial bracket between which roots are searched 
        for. These are the first parameter provided to fun.
    P2 : list of p2 values (floats) of equal length as A, B
        p2 is the second parameter provided to fun. A full bisection is 
        done for each p2 in P2.
    tol : float, optional, defaults to 50
        Tollerance below which |error| as outputed by fun should drop.
    max_iter : int, optional, defaults to 1
        Maximum allowed number of iterations after which the bisection 
        algorithm is stopped and the current results are returned.
    filename : string, optional, defaults to 'bisectionData.txt'
        Text file in which to store the simulation outputs during and 
        after the bisection algorithm. If it does not exist yet it is 
        created.

    Returns
    -------
    midpoints : list
        A0 values found to result in a root for f. If no roots were 
        found the value is set to np.nan.
    uncertainties : list
        Final bracket widths indicating an uncertainty in the midpoint 
        values.
    errors : list
        Final values of f(midpoints) giving the deviation from 0.
    """
    # Turn lists into arrays and preallocate arrays to parameters
    A = np.array(A)
    B = np.array(B)
    P2 = np.array(P2)
    midpoints = np.empty(len(P2))
    fMid = np.empty(len(P2))
    included = np.zeros(len(P2), dtype=bool)
    right_roots = np.zeros(len(P2), dtype=bool)
        
    # Run initial simulations at the specified bracket bounds
    fA = f(A, P2)
    fB = f(B, P2)
        
    # Store intermediate results, open and close file so that it gets 
    # updated immediately
    file = open(filename, mode='a+')
    for a, b, p2, fa, fb in zip(A, B, P2, fA, fB):
        file.write(f"{a:1.5e}, {p2:1.5e}, {fa/1e3+200:1.1f}\n")
        file.write(f"{b:1.5e}, {p2:1.5e}, {fb/1e3+200:1.1f}\n")
    file.close()

    # Check if root exists in interval, else exclude from bisection and write 
    # None results
    included[fA*fB <= 0] = True # this way if f=NaN it will be excluded too
    midpoints[~included] = None; fMid[~included] = None
    for p2, fa, fb in zip(P2[~included], fA[~included], fB[~included]): 
        print(f"\033[96mNo roots found for: p2={p2:1.3e}: f(a)={fa:1.3e}, "
              f"f(b)={fb:1.3e}.\x1b[0m")    

    # Break of bisection of no roots were found at all
    if not any(included): return midpoints, (B-A)/2, fMid

    # Run bissection algorithm for all included cases
    for i in range(max_iter):
        midpoints[included] = (A+B)[included]/2.0
        fMid[included] = f(midpoints[included], P2[included])

        file = open(filename, mode='a+')
        for midpoint, p2, fmid in zip(midpoints[included], P2[included], fMid[included]):
            file.write(f"{midpoint:1.5e}, {p2:1.5e}, {fmid/1e3+200:1.1f}\n")
            print(f"\033[96mSearching... iteration {i+1}, error {fmid:1.2e} at" 
                  f" x={midpoint:1.3e}\x1b[0m")
        file.close()
        
        # Exclude those that are already finished or resulting in DREAM error
        included[np.abs(fMid) < tol] = False
        included[np.isnan(fMid)] = False
        if not any(included): break
        
        # Root between a and midpoint
        left_roots = (fA*fMid < 0)*included
        B[left_roots] = midpoints[left_roots]
        # Root between midpoint and b
        right_roots = (~left_roots)*included
        A[right_roots] = midpoints[right_roots]
        fA[right_roots] = fMid[right_roots]

    return midpoints, (B-A)/2, fMid

def sensitivity_scan(argument_lists,param,scan_list,rel_step_A0=0.1,step_p2=1e-6,Ip=None):
    """
    Computes the relative sensitivity of y to x defined as 
    Syx = x/y*dy/dx, with y=I_p and x=A0 and p2, by doing four DREAM 
    simulations around the given settings slightly varying A0 and p2.
    
    Parameters
    ----------
    argument_lists : 
    param : string
        Name of parameter p2.
    rel_step_A0 : float
        Size of +/- relative step made in A0 for determining S_Ip_A0.
    step_p2 : float
        Size of +/- step made in p2 for determining S_Ip_p2.
    Ip : list, optional, defaults to a list filled with 200e3
        Plasma current at initial settings (before stepping).
    
    Returns
    -------
    S_Ip_A0, S_Ip_p2 : 1d np.ndarray's of same length as argument_lists
        Relative plasma current sensitivities to respectively A0 and p2.
    """
    settings = np.empty(4*len(argument_lists), dtype=object)
    output_names = np.empty(4*len(argument_lists), dtype=object)
    if Ip is None: Ip = np.ones(len(argument_lists))*200e3
    A0 = np.empty(len(argument_lists), dtype=float)
    P2 = scan_list
    S_Ip_A0 = np.empty(len(argument_lists), dtype=float)
    S_Ip_p2 = np.empty(len(argument_lists), dtype=float)
    
    # Generate DREAM settings
    for i, args in enumerate(argument_lists):
        # Step around (A0,p2) in all 4 directions
        extension = args.extension
        A0[i] = args.A0
        args.A0 = A0[i]*(1+rel_step_A0)
        args.extension = f'_{i:1.0f}_A0_plus'+extension
        settings[4*i], output_names[4*i] = SetUp(args).apply_settings(
            verbose=args.verbose, viz=args.visualize)
        settings[4*i].solver.setVerbose(args.verbose)
        
        args.A0 = A0[i]/(1+rel_step_A0)
        args.extension = f'_{i:1.0f}_A0_min'+extension
        settings[4*i+1], output_names[4*i+1] = SetUp(args).apply_settings(
            verbose=args.verbose, viz=args.visualize)
        settings[4*i+1].solver.setVerbose(args.verbose)
        
        args.__dict__[param] = P2[i]+step_p2
        args.extension = f'_{i:1.0f}_{param}_plus'+extension
        settings[4*i+2], output_names[4*i+2] = SetUp(args).apply_settings(
            verbose=args.verbose, viz=args.visualize)
        settings[4*i+2].solver.setVerbose(args.verbose)
        
        args.__dict__[param] = P2[i]-step_p2
        args.extension = f'_{i:1.0f}_{param}_min'+extension
        settings[4*i+3], output_names[4*i+3] = SetUp(args).apply_settings(
            verbose=args.verbose, viz=args.visualize)
        settings[4*i+3].solver.setVerbose(args.verbose)
    
    # Run DREAM disruption simulations
    print("Sensitivity simulations...")
    dos = runiface_parallel(settings, output_names, quiet=args.quiet,
                            njobs=args.Njobs)
    #dos = [DREAMOutput(outname) for outname in output_names]
                            
    # Reshape so dos[i,0] --> 2 A0 scans and dos[i,1] --> p2 scans of args i
    dos = np.array(dos).reshape(len(argument_lists),2,2)
    # Compute plasma current sensitivity from outputs
    for i, do in enumerate(dos):
        try: 
            # Use Ip where RE current is max is start-of-plateau current
            Ire = do[0,0].eqsys.j_re.current()
            Ire_max = Ire[do[0,0].grid.t[:]>1e-4].max() # exclude spike during TQ
            Ipm = do[0,0].eqsys.I_p[:].flatten()[Ire==Ire_max][0]
            do[0,0].close()
            Ire = do[0,1].eqsys.j_re.current()
            Ire_max = Ire[do[0,1].grid.t[:]>1e-4].max() # exclude spike during TQ
            Ipp = do[0,1].eqsys.I_p[:].flatten()[Ire==Ire_max][0]
            do[0,1].close()
            
            # dy/y = (Ipp-Ipm)/Ip; dx/x = [(1+s)*A0-A0/(1+s)]/A0 = (1+s)-1/(1+s)
            S_Ip_A0[i] = (Ipp-Ipm)/((1+rel_step_A0)-1/(1+rel_step_A0))/Ip[i]
        except: 
            S_Ip_A0[i] = np.nan
            print(f"DREAM outputed an error for A0 sensitivity scan simulation labeled {i}")
        
        try: 
            # Use Ip where RE current is max is start-of-plateau current
            Ire = do[1,0].eqsys.j_re.current()
            Ire_max = Ire[do[1,0].grid.t[:]>1e-4].max() # exclude spike during TQ
            Ipm = do[1,0].eqsys.I_p[:].flatten()[Ire==Ire_max][0]
            do[1,0].close()
            Ire = do[1,1].eqsys.j_re.current()
            Ire_max = Ire[do[1,1].grid.t[:]>1e-4].max() # exclude spike during TQ
            Ipp = do[1,1].eqsys.I_p[:].flatten()[Ire==Ire_max][0]
            do[1,1].close()
            
            S_Ip_p2[i] = (Ipp-Ipm)/(2*step_p2)*P2[i]/Ip[i]
        except: 
            S_Ip_p2[i] = np.nan
            print(f"DREAM outputed an error for {param} sensitivity scan simulation labeled {i}")
        
    return A0, S_Ip_A0, S_Ip_p2


def loop(argument_lists):
    """
    Main loop generating DREAM settings objects and running parallel 
    simulations.

    Parameters
    ----------
    argument_lists : list of argparse namespaces
        Each namespace contains all the tunable settings for a DREAM 
        disruption simulation using this script.
    
    Returns
    -------
    Ip : np.ndarray
        Final plasma current(s) as outputed by DREAM simulation(s).
    """
    settings = np.empty(len(argument_lists), dtype=object)
    output_names = np.empty(len(argument_lists), dtype=object)
    Ip = np.empty(len(argument_lists), dtype=float)
    
    # Generate DREAM settings
    for i, args in enumerate(argument_lists):
        settings[i], output_names[i] = SetUp(args).apply_settings(
            verbose=args.verbose, viz=args.visualize)
        settings[i].solver.setVerbose(args.verbose)
    
    # Run DREAM disruption simulations
    print("Disruption simulation...")
    dos = runiface_parallel(settings, output_names, quiet=args.quiet,
                            njobs=args.Njobs)
    
    # Extract plasma current from DREAM outputs
    for i, do in enumerate(dos):
        try: 
            # Use Ip where RE current is max is start-of-plateau current
            Ire = do.eqsys.j_re.current()
            Ire_max = Ire[do.grid.t[:]>1e-4].max() # exclude spike during TQ
            Ip[i] = do.eqsys.I_p[:].flatten()[Ire==Ire_max][0]
            do.close()
        except: 
            Ip[i] = np.nan
            print(f"DREAM outputed an error for simulation labeled {i}")
        
    return Ip

    
def main(argv, param='', unit='', scan='', scan_list=[]):
    """
    Function preparing for and calling the simulation functions.

    Parameters
    ----------
    argv : list
        Command line input arguments.
    param : string, optional
        Which parameter to scan over (if any). Can by any of the 
        arguments in the parser.
    unit : string, optional
        Unit of parameter to scan over (if any), to include in header of
        output file.
    scan : string, optional
        Scan label to add to parameter scan output file (if any).
    scan_list : list, optional
        Values of `param` to include in scan. If `scan_list` is empty,
        it is replaced by [tco] (the cut-off time in magnetic 
        perturbation) and used in the bisection algorithm, or not used 
        at all if no bisection is done.
    
    Returns
    -------
    Ip : np.ndarray
        Final plasma current(s) as outputed by DREAM simulation.
    """

    # Load arguments
    ARGS = parse_args(argv)
    Ip = None
    
    if ARGS.Nthreads: os.environ['OMP_NUM_THREADS'] = str(ARGS.Nthreads)

    write_scan_file = True
    ## If a scan list is provided, scan over the corresponding parameter
    if scan_list: 
        # Prepare list of arguments for all simulations
        argument_lists = np.empty(len(scan_list),dtype=object)
    
        # Prepare scan output file
        filename = f"output/{param}Scan{scan}.txt"
        if not os.path.isfile(filename): write_header = True
        else: write_header = False
        
        for i, p in enumerate(scan_list):
            args = deepcopy(ARGS)
            args.__dict__[param] = p
            args.prefix = ARGS.prefix+f"{param}scan{scan}_"
            # Important so that init files are not overwritten
            args.extension += f'_{i:1.0f}'
            argument_lists[i] = args
    ## If no scan list, bisection made with P2=tco as default, which can be a 
    ## list as well
    else:
        write_scan_file = False
        argument_lists = np.array([deepcopy(ARGS)],dtype=object)
        param = 'tco'
        scan_list = [ARGS.tco]
            
    ## If bisection bounds are provided, run bisection algortithm to find A0 
    ## where Ip_plateau=200kA
    if not None in ARGS.bisect:
        # Bisected function: roots are 0 current deviation from 200kA
        def f(A0,P2):
            # Make sure that when input lists get shorter due to exclusion 
            # (finished or crashed), the correct args are used for the sims 
            # that continue
            included = np.zeros(len(argument_lists), dtype=bool)
            for i, args in enumerate(argument_lists):
                if args.__dict__[param] in P2:
                    included[i] = True
                    args.__dict__['A0'] = A0[P2==args.__dict__[param]][0]
                    
            Ip = loop(argument_lists[included])
            return Ip - 200e3
        
        N = len(scan_list)
        dBB200, udBB, uIp = bisectionParallel(f, [ARGS.bisect[0]]*N, 
            [ARGS.bisect[1]]*N, scan_list, tol=50, max_iter=ARGS.max_iter,
            filename=f'{ARGS.prefix}IpData_{param}{ARGS.extension}.txt')
        Ip = uIp + 200e3

        # Print bisection results
        write_scan_file = False
        for p2, x, ux, eps in zip(scan_list, dBB200, udBB, uIp):
            print(f"\033[96mp2 = {p2:1.3e}: x = {x:1.5e} \u00B1 {ux:1.1e}; "
                  f"Ip-200kA = {eps:1.3f}.\x1b[0m")

    ## If no bisection required, run single or multiple simulation(s)
    else:
        Ip = loop(argument_lists)
        
    ## Sensitivity scan around finished simulation in A0 and p2
    if ARGS.sensitivity_scan:
        A0s, S_Ip_A0, S_Ip_p2 = sensitivity_scan(
            argument_lists, param, scan_list, Ip=Ip)
        file = open(f"{ARGS.prefix}SensitivityData_A0-{param}{ARGS.extension}.txt",mode='a+')
        file.writelines([f"{A0:1.5e}, {p2:1.5e}, {SA:1.5e}, {Sp:1.5e}\n" for
                         (A0,p2,SA,Sp) in zip(A0s,scan_list,S_Ip_A0,S_Ip_p2)])
        file.close()
    
    # Write output of parameter scan if any
    if write_scan_file:
        file = open(filename,mode='a+')
        if write_header: file.write(f"#, {param} ({unit}), Ip (A)\n")
        file.writelines([f"{i:1.0f}, {p:1.5e}, {out}\n" for i,(p,out) in 
                         enumerate(zip(scan_list,Ip))])
        file.close()
    return Ip
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:], ui.param, ui.unit, ui.scan, ui.scan_list))
