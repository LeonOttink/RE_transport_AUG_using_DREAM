import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter
import sys
from scipy.interpolate import griddata, interp1d
import GeriMap

sys.path.append('../py/') # go to local DREAM module location
from DREAM import DREAMOutput, DREAMSettings

## Compute SSF
def survivingSeedFraction(outfile):
    unpert = np.genfromtxt('DATA/SSF/no_perturbation.txt')
    
    do = DREAMOutput(outfile)
    t  = do.grid.t[:]
    dt = np.diff(t)
    R0 = do.grid.R0[0]
    
    # Total number of RE generated
    nre = do.eqsys.n_re[:]
    Nre = do.grid.integrate(nre)*R0
    
    # Primary RE generated
    dnredt_pri = -do.other.fluid.gammaFhot[:]
    nre_pri = np.cumsum(dnredt_pri*dt[:,np.newaxis],axis=0)
    Nre_pri = do.grid.integrate(nre_pri)*R0
            
    # Avalanche generated RE
    dnredt_ava = do.other.fluid.GammaAva[:]*nre[1:]
    nre_ava  = np.cumsum(dnredt_ava*dt[:,np.newaxis],axis=0)
    Nre_ava = do.grid.integrate(nre_ava)*R0
    
    # Total, primary and avalanche RE transported out of plasma edge
    Lr = -do.other.scalar.radialloss_n_re[:].flatten()*R0
    Ntp = np.cumsum(Lr*dt)
    Ntp_pri = np.cumsum(Lr*(nre[1:,-1]-nre_ava[:,-1])/nre[1:,-1]*dt)
    Ntp_ava = np.cumsum(Lr*nre_ava[:,-1]/nre[1:,-1]*dt)
    
    # SSF
    SSF1 = (Nre_pri[-1]+Ntp[-1])/unpert[-1,2]
    itco = np.where(Nre==Nre[t>20e-6].min())[0][0]
    SSF2 = Nre[itco]/unpert[-1,2]
    SSF3 = Nre[t<=1e-4][-1]/unpert[-1,2]
    SSF4 = (Nre[itco]+Nre_pri[-1]-Nre_pri[itco])/unpert[-1,2]
    SSF5 = (Nre_pri[-1]+Ntp_pri[-1])/unpert[-1,2]
    SREF = Nre[-1]/unpert[-1,1]
    avatco = Nre_ava[itco]/unpert[-1,1]
    Ipf = do.eqsys.I_p[:].flatten()[-1]
    print(f"SSF = {SSF4*1000:1.2f}‰")
    #np.savetxt('output/SSF/no_perturbation.txt',np.array([t[1:],Nre[1:],Nre_pri+Ntp,Nre_ava]).T)
    
    if False:
        fig, ax = plt.subplots(1,figsize=(5,3))
        ax.plot(unpert[:,0],unpert[:,1],c='C1',ls='--',label='Total w/o transport')
        ax.plot(unpert[:,0],unpert[:,2],c='C0',ls='--',label='Seed w/o transport')
        ax.plot(t[:],Nre[:],c='C1',label=r'Total w/ transport')
        ax.set_xlabel("$t$ (s)")
        ax.set_ylabel("$N_{RE}$ (#)")
        ax.legend()
        plt.tight_layout()
        plt.savefig('SSF_method.pdf')
        plt.show()

    return SSF1,SSF2,SSF3,SSF4,SSF5,SREF,avatco,Ipf

def SSF_loop(path='DATA/dBB/constant/'):
    data = [{'filename': 'tcoscan0_output_0.h5', 'tco': 35e-6, 'A0': 9.28027e-03},
            {'filename': 'tcoscan0_output_1.h5', 'tco': 40e-6, 'A0': 7.41797e-03},
            {'filename': 'tcoscan0_output_2.h5', 'tco': 45e-6, 'A0': 6.14453e-03},
            {'filename': 'tcoscan0_output_3.h5', 'tco': 50e-6, 'A0': 5.26123e-03},
            {'filename': 'tcoscan0_output_4.h5', 'tco': 55e-6, 'A0': 4.62354e-03},
            {'filename': 'tcoscan0_output_5.h5', 'tco': 60e-6, 'A0': 4.14453e-03},
            {'filename': 'tcoscan0_output_6.h5', 'tco': 65e-6, 'A0': 3.77246e-03},
            {'filename': 'tcoscan0_output_7.h5', 'tco': 70e-6, 'A0': 3.47510e-03},
            {'filename': 'tcoscan0_output_8.h5', 'tco': 75e-6, 'A0': 3.23120e-03},
            {'filename': 'tcoscan0_output_9.h5', 'tco': 80e-6, 'A0': 3.02734e-03},
            {'filename': 'tcoscan0_output_10.h5', 'tco': 85e-6, 'A0': 2.85376e-03}]

    with open(path+'SSF_data.txt','w') as f:
        f.write('tco (s), A0 (#), SSF -full tp (lower), SSF nRE at tco, SSF nRE at 100us, SSF nRE at tco+pri.gen.rest., SSF -partial tp (higher), SREF nRE at final, ava w/o tp at tco, Ipf\n')
        for i,dat in enumerate(data):
            SF = survivingSeedFraction(path+dat['filename'])
            f.write(f"{dat['tco']:1.2e}, {dat['A0']:1.5e}, {SF[0]:1.5e}, {SF[1]:1.5e}, {SF[2]:1.5e}, {SF[3]:1.5e}, {SF[4]:1.5e}, {SF[5]:1.5e}, {SF[6]:1.5e}, {SF[7]:1.7e}\n")


## dBB plot
def __getContourGrid__(xi,yi,zi, nx=100, ny=100,log=False):
    interpmode = 'linear'
    if log: 
        # Set up a regular grid of interpolation points
        x = np.logspace(np.log10(xi.min()), np.log10(xi.max()), nx)
        y = np.logspace(np.log10(yi.min()), np.log10(yi.max()), ny)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate
        z = griddata(np.array([np.log10(xi),np.log10(yi)]).T, zi,
            np.array([np.log10(X).flatten(),np.log10(Y).flatten()]).T, 
            method=interpmode)
        Z = z.reshape((ny,nx))
    
    else: 
        # Set up a regular grid of interpolation points
        x = np.linspace(xi.min(), xi.max(), nx)
        y = np.linspace(yi.min(), yi.max(), ny)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate
        z = griddata(np.array([xi,yi]).T, zi,
            np.array([X.flatten(),Y.flatten()]).T, 
            method=interpmode)
        Z = z.reshape((ny,nx))
        
    return X, Y, Z

def dBB_parameterPlot2D(filename="DATA/dBB/IpData_constant.txt"):
    data = np.genfromtxt(filename,delimiter=', ',skip_header=1)
    data = np.array(sorted(data, key=lambda element: (element[0], element[1], element[2])))
    Ip  = data[:,2]
    tco = data[:,1]*1e6
    A0  = data[:,0]

    log = True
    X, Y, Z = __getContourGrid__(A0,tco,Ip,log=log)

    fig, ax = plt.subplots(1)
    im = ax.contourf(X, Y, Z, cmap='GeriMap',levels=np.linspace(0,800,101))
    ax.scatter(A0[Ip!=200.0], tco[Ip!=200.0], c=Ip[Ip!=200.0], cmap='GeriMap',vmin=0,vmax=800, edgecolors='#888888',s=10,linewidths=1)
    ax.scatter(A0[Ip==200.0], tco[Ip==200.0], c='#ffffff',s=15)
    cs = ax.contour(X, Y, Z, [200.0],colors=['#ffffff'])
    
    #p, a, b, c = -0.6, 2, 0,0#-1e-3, 0
    #plt.plot(A0,a*(A0+b)**p,c='green')

    ax.set_xlabel("$(\delta B/B)_0$")
    ax.set_ylabel(r"$t_{\rm cutoff}$ (µs)")
    cbar = fig.colorbar(im,label="$I_{p,f}$ (kA)")
    cbar.set_ticks(ticks=list(np.arange(0,801,100,dtype=int)),labels=list(np.arange(0,801,100,dtype=int)))
    #ax.clabel(cs,cs.levels,inline=True,fontsize=8,rightside_up=True)
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
        ax.set_xlim(2e-3,1e-2)
#        ax.set_ylim(1e-1,1e2)

    for a in im.collections:
        a.set_edgecolor("face")
    plt.savefig('dBB_profile_parameter_plot_nAr83e18_b250_loglog.pdf')
    plt.show()
    
    
## (NUMERICAL) PARAMETER SCANS
def __plot_A0__(fig,ax,filename,tco=0):
    data = np.genfromtxt(filename,delimiter=', ',skip_header=1)
    data = np.array(sorted(data, key=lambda element: (element[0], element[1], element[2])))

    Ip  = data[data[:,1]==tco,2]
    A0  = data[data[:,1]==tco,0]
    
    ax.plot(A0,Ip,'-o')
    ax.set_xlabel("$(\delta B/B)_0$")
    ax.set_ylabel("$I_{p,f}$ (kA)")

def __plot_tco__(fig,ax,filename="output/scans_dBB-profile/IpDataTimeCutoff.txt",A0=0):
    data = np.genfromtxt(filename,delimiter=', ',skip_header=1)
    data = np.array(sorted(data, key=lambda element: (element[0], element[1], element[2])))
    
    eps = 0.
    i = (data[:,0]>=(1-eps)*A0) * (data[:,0]<=(1+eps)*A0)
    Ip  = data[i,2]
    tco = data[i,1]
    
    ax.plot(tco,Ip,'-o')
    ax.set_xlabel("$t_{cutoff}$ ($\mathrm{\mu}$s)")
    ax.set_ylabel("$I_{p,f}$ (kA)")

def __plot_nAr__(fig,ax,Ip,nAr):
    ax.plot(nAr,Ip/1e3,'-o')
    ax.set_ylabel("$I_{p,f}$ (kA)")
    ax.set_xlabel(r"$n_{Ar}$ (m$^{-3}$)")
    
def __plot_sigma__(fig,ax,Ip,sigma):
    ax.plot(sigma,Ip/1e3,'-o')
    ax.set_ylabel("$I_{p,f}$ (kA)")
    ax.set_xlabel("Argon density profile peakedness (#)")
    
def __plot_pMax__(fig,ax,Ip,pMax):
    ax.plot(pMax,Ip/1e3,'-o')
    ax.set_ylabel("$I_{p,f}$ (kA)")
    ax.set_xlabel(r"$p_{max}$ (m$_e$c)")
    
def __plot_reltol__(fig,ax,Ip,reltol):
    ax.loglog(reltol[:-1],np.abs(Ip[:-1]-Ip[-1])/Ip[-1]*100,'-o')
    ax.set_ylabel("Relative error in $I_{p,f}$ (%)")
    ax.set_xlabel("Relative solver tolerance (#)")
    if len(fig.axes) == 2: 
        ax.invert_xaxis()
        for axi in fig.axes[1:]: axi.remove()
    ax2 = ax.twinx()
    mn, mx = ax.get_ylim()
    ax2.set_ylim(mn*Ip[-1]/100, mx*Ip[-1]/100)
    ax2.set_ylabel("Absolute error in $I_{p,f}$ (A)")
    ax2.set_yscale('log')
    
def __plot_Np__(fig,ax,Ip,Np):
    ax.loglog(Np[:-1],np.abs(Ip[:-1]-Ip[-1])/Ip[-1]*100,'-o')
    ax.set_ylabel("Relative error in $I_{p,f}$ (%)")
    ax.set_xlabel("Number of momentum grid points $N_p$ (#)")
    if len(fig.axes) == 2: 
        for axi in fig.axes[1:]: axi.remove()
    ax2 = ax.twinx()
    mn, mx = ax.get_ylim()
    ax2.set_ylim(mn*Ip[-1]/100, mx*Ip[-1]/100)
    ax2.set_ylabel("Absolute error in $I_{p,f}$ (A)")
    ax2.set_yscale('log')
    
def __plot_Nr__(fig,ax,Ip,Nr):
    ax.semilogx(Nr,Ip/1e3,'-o')
    ax.set_ylabel("$I_{p,f}$ (kA)")
    ax.set_xlabel(r"Number of radial grid points $N_r$ (#)")
    
def __plot_dt0__(fig,ax,Ip,dt0):
    ax.semilogx(dt0,Ip/1e3,'-o')
    ax.set_ylabel("$I_{p,f}$ (kA)")
    ax.set_xlabel("Initial time step $dt_0$ (s)")
    ax.invert_xaxis()

def __plot_dtMax__(fig,ax,Ip,dtMax):
    ax.semilogx(dtMax,Ip/1e3,'-o')
    ax.set_ylabel("$I_{p,f}$ (kA)")
    ax.set_xlabel("Maximum time step $dt_{Max}$ (s)")
    ax.invert_xaxis()
    
def __plot_tauwall__(fig,ax,Ip,tauwall):
    xlim = (np.nanmin(tauwall)/10,np.nanmax(tauwall)*100)
    tauwall[np.isnan(tauwall)] = xlim[1]*10
    ax.semilogx(tauwall,Ip/1e3,'-o')
    ax.set_xlim(*xlim)
    ax.set_ylabel("$I_{p,f}$ (kA)")
    ax.set_xlabel(r"$\tau_{wall}$ (s)")
    
def parameterScanPlot(fig,ax,filename):
    header = np.genfromtxt(filename,delimiter=', ',dtype=str,max_rows=1,comments='!')
    data   = np.genfromtxt(filename,delimiter=', ',skip_header=1,comments='!')
    param, unit = header[1].split(' ') # "param (unit)"
    Ip = data[:,2]
    p  = data[:,1]
    
    if   param == 'nAr':    __plot_nAr__(fig,ax,Ip,p)
    elif param == 'sigma':  __plot_sigma__(fig,ax,Ip,p)
    elif param == 'pMax':   __plot_pMax__(fig,ax,Ip,p)
    elif param == 'reltol': __plot_reltol__(fig,ax,Ip,p)
    elif param == 'Np':     __plot_Np__(fig,ax,Ip,p)
    elif param == 'Nr':     __plot_Nr__(fig,ax,Ip,p)
    elif param == 'dt0':    __plot_dt0__(fig,ax,Ip,p)
    elif param == 'dtMax':  __plot_dtMax__(fig,ax,Ip,p)
    
    elif param == 'tauwall':__plot_tauwall__(fig,ax,Ip,p)
    else: pass

    
## Plots from report
def dBB_plot_FULL(filename_constant='DATA/dBB/IpData_constant.txt',filename_edge='DATA/dBB/IpData_edge.txt'):
    fig, axs = plt.subplots(1,2,figsize=(9.3,4),gridspec_kw={'width_ratios':[1,1.2]})
    
    for filename, ax in zip([filename_constant,filename_edge],axs):
        data = np.genfromtxt(filename,delimiter=', ',skip_header=1)
        data = np.array(sorted(data, key=lambda element: (element[0], element[1], element[2])))
        Ip  = data[:,2]
        tco = data[:,1]*1e6
        A0  = data[:,0]

        log = True
        X, Y, Z = __getContourGrid__(A0,tco,Ip,log=log)

        im = ax.contourf(X, Y, Z, cmap='GeriMap',levels=np.linspace(0,800,101))
        ax.scatter(A0[Ip!=200.0], tco[Ip!=200.0], c=Ip[Ip!=200.0], cmap='GeriMap',vmin=0,vmax=800, edgecolors='#888888',s=10,linewidths=1)
        ax.scatter(A0[Ip==200.0], tco[Ip==200.0], c='#ffffff',s=15)
        cs = ax.contour(X, Y, Z, [200.0],colors=['#ffffff'])

        ax.set_xlabel("$(\delta B/B)_0$")
        ax.set_ylabel(r"$t_{\rm cutoff}$ (µs)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
        ax.set_xlim(2e-3,1e-2)

        for a in im.collections:
            a.set_edgecolor("face")
            
    axs[0].set_title(r"$\bf{(a)}$                          Uniform",loc='left',fontsize=10)
    axs[1].set_title(r"$\bf{(b)}$                      Edge localized",loc='left',fontsize=10)
            
    cbar = fig.colorbar(im,label=r"$I_{\rm p,plateau}$ (kA)")
    cbar.set_ticks(ticks=list(np.arange(0,801,100,dtype=int)),labels=list(np.arange(0,801,100,dtype=int)))
    plt.tight_layout()
    plt.savefig('dBB_parameter_plots.pdf')
    plt.show()
    
def SSF_plot_FULL(outfile='DATA/dBB/constant/tcoscan0_output_9.h5',SFF_file='DATA/SSF/SSF_data.txt'):
    unpert = np.genfromtxt('DATA/SSF/no_perturbation.txt')
    do = DREAMOutput(outfile)
    Nre = do.grid.integrate(do.eqsys.n_re[:])*do.grid.R0[0]
    
    data = np.genfromtxt(SFF_file,delimiter=', ',skip_header=1)
    tco = data[:,0]
    A0  = data[:,1]
    SSF4 = data[:,5] # SSF nRE at tco+pri.gen.rest. (higher if ava.gen.>ava.tp. which is always?)
    AVAg = data[:,8] # generated ava Nre at tco
    
    fig, ax = plt.subplots(1,2,figsize=(10.5,3.6),gridspec_kw={'width_ratios':[1,1.4]})
    
    ax[0].plot(unpert[:,0],unpert[:,1]*1e-17,c='C1',ls='--',label='Total w/o transport')
    ax[0].plot(unpert[:,0],unpert[:,2]*1e-17,c='C0',ls='--',label='Seed w/o transport')
    ax[0].plot(do.grid.t[:],Nre[:]*1e-17,c='C1',label=r'Total w/ transport')
    ax[0].set_xlabel("$t$ (s)")
    ax[0].set_ylabel("$N_{RE}$ (10$^{17}$)")
    ax[0].legend()
    
    axt = ax[1].twiny()
    ax[1].semilogx(A0*1e3,AVAg*1e2+3.67,'-o',c='grey',ls='--',label=r"$N_{\mathrm{RE,ava}}(t_{\mathrm{cutoff}})$ (a.u.)")
    ax[1].semilogx(A0*1e3,SSF4*1e4,'-o',c='C0',label="SSF")
    axt.semilogx(A0*1e3,SSF4*1e4,alpha=0,c='C0')
    
    ax[1].set_ylim(3.5,6)
    ax[1].legend(reverse=True)
    ax[1].minorticks_off()
    axt.minorticks_off()
    ax[1].set_xticks(A0*1e3,[f'{a:1.1f}' for a in A0*1e3])    
    axt.set_xticks(A0*1e3,[f'{t:1.0f}' for t in tco*1e6])
    ax[1].xaxis.grid(color='grey',ls=':',alpha=0.8)
    ax[1].set_xlabel(r"$\left(\delta B/B\right)_0$ (10$^{-3}$)")
    axt.set_xlabel(r"$t_{\rm cutoff}$ ($\rm \mu$s)")
    ax[1].set_ylabel("Surviving seed fraction (10$^{-4}$)")
    
    plt.suptitle(r"$\bf{(a)}$                                                                                                          $\bf{(b)}$",fontsize=9,horizontalalignment='left',x=0.04,y=0.89)
    
    plt.tight_layout()
    plt.savefig('SSF.pdf')
    plt.show()

def dBB_sensitivity(filename='DATA/sensitivity/SensitivityData.txt'):
    dataS = np.genfromtxt(filename,delimiter=', ',skip_header=1)
    dataS = np.array(sorted(dataS, key=lambda element: (element[0], element[1])))
    A0  = dataS[:,0]
    tco = dataS[:,1]
    S_Ip_A0  = dataS[:,2]
    S_Ip_tco = dataS[:,3]
    S_A0_tco = (A0[2:]-A0[:-2])/A0[1:-1]*tco[1:-1]/(tco[2:]-tco[:-2])
    
    fig, axA = plt.subplots(1,figsize=(7,4))
    axt = axA.twiny()
    axAt = axA.twinx()
    
    lA = axA.semilogx(A0*1e3,1/S_Ip_A0,'-o',label=r'$\left(\delta B/B\right)_0$ to $I_{\rm p}$')
    lt = axt.semilogx(A0*1e3,1/S_Ip_tco,'-o',c='C1',label=r'$t_{\rm cutoff}$ to $I_{\rm p}$')
    lAt = axAt.semilogx(A0[1:-1]*1e3,S_A0_tco,'-o',c='C2',label=r'$\left(\delta B/B\right)_0$ to $t_{\rm cutoff}$')
    axA.set_xlabel(r"$\left(\delta B/B\right)_0$ (10$^{-3}$)")
    axt.set_xlabel(r"$t_{\rm cutoff}$ ($\rm\mu$s)")
    axA.set_ylabel("Relative sensitivity to plateau current (#)")
    axAt.set_ylabel("Relative sensitivity to cut-off time (#)",labelpad=10)

    axA.minorticks_off()
    axt.minorticks_off()
    axA.set_xticks(A0*1e3,[f'{a:1.1f}' for a in A0*1e3])    
    axt.set_xticks(A0*1e3,[f'{t:1.0f}' for t in tco*1e6])
    axAt.yaxis.label.set_color('C2')
    axAt.spines["right"].set_edgecolor('C2')
    axAt.tick_params(axis='y', colors='C2')
    
    lns = lA+lt+lAt
    plt.legend(lns,[l.get_label() for l in lns],loc='center left')
    axA.xaxis.grid(color='grey',ls=':',alpha=0.8)
    plt.savefig('dBB_sensitivity.pdf')
    plt.show()

def radial_density_influence_FULL():
    path = "DATA/density_influence/output_"
    dos = [DREAMOutput(path+"constant.h5"),DREAMOutput(path+"edge.h5"),DREAMOutput(path+"constant_A3D3.h5"),DREAMOutput(path+"edge_A3D3.h5")]
    cmap = plt.get_cmap('GeriMap')
    times = [50,55,60,65,70,75,80,-1]
    mask = lambda a,b,t: a if ti not in t else b
    
    fig, ax = plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True)
    ax = ax.flatten()
    for j,do in enumerate(dos):
        for i,ti in enumerate(times):
            if j==1: t=[65,75,-1]; 
            else: t=[65,-1]
            label = fr"$t=${do.grid.t[ti]*1e6:1.0f} $\rm\mu$s"
            if ti==-1: label = fr"$t=${do.grid.t[ti]*1e3:1.0f} ms"
            ax[j].plot(do.grid.r[:]/0.5, do.eqsys.n_re[ti,:]/do.eqsys.n_re[ti,:].max(), c=cmap(i/len(times)),
                       label=label,alpha=mask(0.3,1,t),ls=mask('--','-',t),lw=mask(1.5,2.5,t))
    fig.supylabel(r"$n_{\rm RE}/\max{(n_{\rm RE})}$",x=0.06)
    fig.supxlabel("$r/a$")
    fig.subplots_adjust(wspace=0.08,hspace=0.2)
    ax[1].legend(loc='center right')
    ax[0].set_title(r"$\bf{(a)}$         Uniform, Diffusion dominated",loc='left')
    ax[1].set_title(r"$\bf{(b)}$    Edge localized, Diffusion dominated",loc='left')
    ax[2].set_title(r"$\bf{(c)}$        Uniform, Advection dominated",loc='left')
    ax[3].set_title(r"$\bf{(d)}$   Edge localized, Advection dominated",loc='left')
    plt.savefig("density_profile_evolution.pdf")
    plt.show()

## CALL SELECTED PLOT FUNCTION
def MAIN(argv):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#000000', '#0572f0', '#ff8103', '#13bd2a', '#f70227', 
                                                        '#f7cf19','#c719f7','#27d6d0'])
    if argv[0] == 'dBB': dBB_parameterPlot2D(argv[1])
    elif argv[0] == 'dBBfull': dBB_plot_FULL(argv[1],argv[2])
    elif argv[0] == 'SSFfull': SSF_plot_FULL(argv[1],argv[2])
    elif argv[0] == 'sens': dBB_sensitivity(argv[1])
    elif argv[0] == 'dens': radial_density_influence_FULL()    

    else: print("The first commandline argument should be of:"
                "dBB, dBBfull, SSFfull, sens or dens")
    
if __name__ == '__main__':
    sys.exit(MAIN(sys.argv[1:]))
