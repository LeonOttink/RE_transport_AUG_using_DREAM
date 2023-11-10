import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter
import sys
from scipy.interpolate import griddata, interp1d
import GeriMap
#GeriMap.register()

sys.path.append('../py/') # go to local DREAM module location
from DREAM import DREAMOutput, DREAMSettings

## MAIN DATA PLOTS
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

def dBB_parameterPlot2D(filename="output/scans_dBB-profile/IpDataTimeCutoff.txt"):
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
    ax.set_ylabel("$t_{cut off}$ (µs)")
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
    
def dBB_sensitivityPlot(filename="output/scans_dBB-profile/IpDataTimeCutoff.txt"):
    data = np.genfromtxt(filename,delimiter=', ',skip_header=1)
    
    # A0 sens
    data = np.array(sorted(data, key=lambda element: (element[1], element[0], element[2])))
    ic = np.where(np.diff(data[:,1],prepend=np.nan))[0]
    szx = []; xx = []; yx = []

    ic = np.append(ic,ic[-1]+1)
    for i in range(ic.size-1):
        x = data[:,0][ic[i]:ic[i+1]]
        y = data[:,1][ic[i]:ic[i+1]]
        z = data[:,2][ic[i]:ic[i+1]]
        
        xx += list((x[1:]+x[:-1])/2)
        yx += list(y[:-1])
        szx += list((z[1:]-z[:-1])/(x[1:]-x[:-1]) * (x[1:]+x[:-1])/(z[1:]+z[:-1]))

    # tco sens !!not usefull with this data!
    data = np.array(sorted(data, key=lambda element: (element[0], element[1], element[2])))
    ic = np.where(np.diff(data[:,0],prepend=np.nan))[0]
    szy = []; xy = []; yy = []

    ic = np.append(ic,ic[-1]+1)
    for i in range(ic.size-1):
        x = data[:,0][ic[i]:ic[i+1]]
        y = data[:,1][ic[i]:ic[i+1]]
        z = data[:,2][ic[i]:ic[i+1]]
        
        xy += list(x[:-1])
        yy += list((y[1:]+y[:-1])/2)
        szy += list((z[1:]-z[:-1])/(y[1:]-y[:-1]) * (y[1:]+y[:-1])/(z[1:]+z[:-1]))
    
    log = True
    X,  Y,  Z   = __getContourGrid__(data[:,0],data[:,1],data[:,2],log=log)
    Xx, Yx, Szx = __getContourGrid__(np.array(xx),np.array(yx),np.array(szx),log=log)
    Xy, Yy, Szy = __getContourGrid__(np.array(xy),np.array(yy),np.array(szy),log=log)

    fig, ax = plt.subplots(1,3,sharey=True)
    im  = ax[0].contourf(X, Y, Z, cmap='GeriMap',levels=np.linspace(0,np.nanmax(Z),100))
    imA = ax[1].contourf(Xx, Yx, Szx, cmap='GeriMap',levels=100,vmax=0.)
    imt = ax[2].contourf(Xy, Yy, Szy, cmap='GeriMap',levels=100,vmax=0.)
    
    fig.colorbar(im, ax=ax[0],label="$I_{p,f}$ (kA)")
    fig.colorbar(imA,ax=ax[1],label="Normalized sensitivity to $(\delta B/B)_0$")
    fig.colorbar(imt,ax=ax[2],label="Normalized sensitivity to $t_{cutoff}$")
    
    ax[0].scatter(data[:,0][data[:,2]!=200.0], data[:,1][data[:,2]!=200.0], c=data[:,2][data[:,2]!=200.0],
                  cmap='GeriMap',edgecolors='#888888',s=10,linewidths=1)
    ax[1].scatter(xx, yx, c=szx, cmap='GeriMap', edgecolors='#888888',s=10,linewidths=1)
    ax[2].scatter(xy, yy, c=szy, cmap='GeriMap', edgecolors='#888888',s=10,linewidths=1)
    ax[0].scatter(data[:,0][data[:,2]==200.0], data[:,1][data[:,2]==200.0], c='#ffffff',s=15)

    ax[0].set_ylabel("$t_{cut off}$ (µs)")
    for x in ax:
        x.contour(X, Y, Z, [200.0],colors=['#ffffff'])
        x.set_xlabel("$(\delta B/B)_0$")
        if log:
            x.set_xscale('log')
            x.set_yscale('log')
    plt.show()
    
    
## (NUMERICAL) PARAMETER SCANS
def __plot_A0__(fig,ax,filename="output/scans_dBB-profile/IpDataTimeCutoff.txt",tco=0):
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
    
def parameterScanPlot(fig,ax,filename="output/Scan.txt"):
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
    
def detective():
    check = 'dt0'

    if check == 'dt0':
        keys = ['2e-14','2e-13','2e-12','2e-11']
        dofs = {key: DREAMOutput(f'output/scans_dt0/dt0scan1_output_{key}.h5') for key in keys}
        keys.append('Const 2e-8'); keys.append('Const 2e-9'); keys.append('Const 2e-10'); 
        keys.append('Const 2e-11'); keys.append('Const 2e-12')
        dofs['Const 2e-8'] = DREAMOutput(f'output/scans_dt0/dt0scan5_output_2e-8.h5')
        dofs['Const 2e-9'] = DREAMOutput(f'output/scans_dt0/dt0scan5_output_2e-9.h5')
        dofs['Const 2e-10'] = DREAMOutput(f'output/scans_dt0/dt0scan5_output_2e-10.h5')
        dofs['Const 2e-11'] = DREAMOutput(f'output/scans_dt0/dt0scan5_output_2e-11.h5')
        dofs['Const 2e-12'] = DREAMOutput(f'output/scans_dt0/dt0scan2_output_2e-12.h5')
    elif check == 'Np':
        keys = [53,80,120,180,270,405,607]
        dofs = {key: DREAMOutput(f'output/scans_N/Npscan2_output_{key}.h5') for key in keys}
    elif check == 'pMax':
        keys = ['07','1','15','2','25']
        dofs = {key: DREAMOutput(f'output/scans_pMax/pMaxscan3_output_{key}.h5') for key in keys}
    
    Efield = 0
    Tcold  = 1
    nre    = 2
    RErate = None
    nAr    = None
    Emax   = None
    fhot   = None
    IpRE   = None
    dt     = 3
    
    lw = [None]*len(keys)
    ls = ['-','-','-','-','-','--','--','--','--']
    c = [None]*len(keys)
    
    fig, ax = plt.subplots(2,2,sharex=False)
    ax = np.array(ax).T.flatten()
    
    if isinstance(Efield,int) and Efield < ax.size:
        ax[Efield].set_xlabel("$t$ (s)")
        ax[Efield].set_ylabel(r"$\left(E/E_{C,eff}\right)(r=0)$ (#)")
        for j,key in enumerate(np.flip(keys)):
            Es = dofs[key].eqsys.E_field[1:,0]/dofs[key].other.fluid.Eceff[:,0]
            ax[Efield].semilogx(dofs[key].grid.t[1:],Es,label=key,lw=lw[j],ls=ls[j],c=c[j])
        ax[Efield].legend()
                            
    if isinstance(Tcold,int) and Tcold < ax.size:
        ax[Tcold].set_xlabel("$t$ (s)")
        ax[Tcold].set_ylabel(r"$T_{cold}(r=0)$ (eV)")
        for j,key in enumerate(np.flip(keys)):
            ax[Tcold].loglog(dofs[key].grid.t[:],dofs[key].eqsys.T_cold[:,0],label=key,lw=lw[j],ls=ls[j],c=c[j])
        
    if isinstance(nre,int) and nre < ax.size:
        ax[nre].set_xlabel("$t$ (s)")
        ax[nre].set_ylabel(r"$n_{RE}(r=0)$ (m$^{-3}$)")
        for j,key in enumerate(np.flip(keys)):
            ax[nre].plot(dofs[key].grid.t[:],dofs[key].eqsys.n_re[:,0],lw=lw[j],ls=ls[j],c=c[j])
        
    if isinstance(RErate,int) and RErate < ax.size:
        ax[RErate].set_xlabel("$t$ (s)")
        ax[RErate].set_ylabel(r"$dn_{RE}/dt(r=0)$ (m$^{-3}/s$)")
        for j,key in enumerate(np.flip(keys)):
            ax[RErate].plot(dofs[key].grid.t[1:],dofs[key].other.fluid.runawayRate[:,0],lw=lw[j],ls=ls[j],c=c[j])
    
    if isinstance(IpRE,int) and IpRE < ax.size:
        ax[IpRE].set_xlabel("$t$ (s)")
        ax[IpRE].set_ylabel(r"$I_{RE}$ (kA)")
        for j,key in enumerate(np.flip(keys)):
            ax[IpRE].plot(dofs[key].grid.t[:],dofs[key].eqsys.j_re.current()[:]/1e3,lw=lw[j],ls=ls[j],c=c[j])
                       
    if isinstance(dt,int) and dt < ax.size:
        ax[dt].set_xlabel("$t$ (s)")
        ax[dt].set_ylabel(r"$dt$ (s)")
        for j,key in enumerate(np.flip(keys)):
            ax[dt].loglog((dofs[key].grid.t[1:]+dofs[key].grid.t[:-1])/2,np.diff(dofs[key].grid.t[:]),
                          lw=lw[j],ls=ls[j],c=c[j])
                         
    if isinstance(nAr,int) and nAr < ax.size:
        ax[nAr].set_xlabel("$t$ (s)")
        ax[nAr].set_ylabel(r"$n_{Ar}/dt(r=0)$ (m$^{-3}/s$)")
        for j,key in enumerate(np.flip(keys)):
            for i,n_i in enumerate(dofs[key].eqsys.n_i['Ar']):
                ax[nAr].semilogx(dofs[key].grid.t[:],n_i[:,0],label=key+f', Z={i}',c=f'C{i}',
                                 lw=[3,3,1,1,1,1,1][j],ls=['-',':','--','-.'][j%4])
        ax[nAr].legend()
        
    if isinstance(Emax,int) and Emax < ax.size:
        ax[Emax].set_xlabel("$t$ (s)")
        ax[Emax].set_ylabel(r"Maximum available energy (?)")
        for j,key in enumerate(np.flip(keys)):
            t = len(dofs[key].grid.t[:])
            N = 1 if j!=0 else 100
            ax[Emax].loglog(dofs[key].grid.t[::N],dofs[key].eqsys.E_field.maxEnergy(t=range(0,t,N))[:,0])
    
    if isinstance(fhot,int) and fhot < ax.size:
        ax[fhot].set_xlabel("$p$ (mc)")
        ax[fhot].set_ylabel(r"$f_{hot}(r=0)$ (?)")

        for j,key in enumerate(np.flip(keys)):
            t = dofs[key].grid.t[:]
            for i, t0 in enumerate([4.2e-7,0.99e-6,1e-5,1e-4]):
                it = np.abs(t-t0).argmin()
                f = dofs[key].eqsys.f_hot[it,0,0,:] # t, r, xi, p
                ax[fhot].semilogy(dofs[key].grid.hottail.p[:],f[:],label=key+f', t={t[it]:1.2e}',
                                  ls=['-','--',':','-.'][i],c=f'C{j}')
        ax[fhot].legend()
    
    plt.tight_layout()
    plt.show()


## SURVIVING SEED FRACTION
"""
def survivingSeedFraction(outfile,setfile):
    unpert = np.genfromtxt('output/SSF/no_perturbation.txt')
    
    do = DREAMOutput(outfile)
    ds = DREAMSettings(setfile)
    t  = do.grid.t[:]
    dt = np.diff(t)
    r  = do.grid.r[:]
    r_f  = do.grid.r_f[:]
    dr = do.grid.dr[0]
    R0 = do.grid.R0[0]
    Vp = do.grid.VpVol[:]*R0
    Ar  = ds.eqsys.n_re.transport.ar[:]
    Drr = ds.eqsys.n_re.transport.drr[:]
    transport_r = ds.eqsys.n_re.transport.todict()['drr']['r']
    transport_t = ds.eqsys.n_re.transport.todict()['drr']['t']
    Phihot = -do.other.fluid.gammaFhot[:]
    
    # Total number of RE generated and transported away
    Lr = -do.other.scalar.radialloss_n_re[:].flatten()*R0
    nre = do.eqsys.n_re[:]
    Nre = do.grid.integrate(nre)*R0
    Ntp = np.cumsum(Lr*dt)
    print(f'{Nre[0]:1.3e}')
    # Primary and avalanche
    A = interp1d(transport_r,interp1d(transport_t,Ar[:,:],axis=0,fill_value="extrapolate")(t[1:]),axis=1,fill_value="extrapolate")
    D = interp1d(transport_r,interp1d(transport_t,Drr[:,:],axis=0,fill_value="extrapolate")(t[1:]),axis=1,fill_value="extrapolate")
    
    
    ddr = lambda X: np.interp(r_f[1:-1],np.diff(X,axis=-1)/dr,axis=-1,fill_value="extrapolate")(r)
    Trnsp = lambda n: 1/Vp * ddr(Vp * (A*n + D*ddr(n)))
    for l in range(0,len(t)-1):
        A(r[i])[l],D(r[i])[l], n[l,i]
        Vp[i], dr
        dt[l]
        
        nseed[l+1] = nseed[l] + dt[l] * (Phihot[l] + Trnsp(nseed[l]))

        
        

    
    plt.figure()
    plt.plot(Nre)
    plt.plot(Nre_ava+Nre_pri)
    plt.show()
    
    # SSF
    surviving_seed = Nre_pri[-1]+Ntp_pri[-1]
    unpert_seed = unpert[-1,1]
    print(f"SSF = {surviving_seed/unpert_seed*100:1.2f}%")

    fig, ax = plt.subplots(1)
    ax.plot(t[:],Nre[:],label=r'total ($n_{\rm RE}$)')
    ax.plot(t[1:],Nre_pri+Ntp_pri,label=r'seed ($\Phi_{\rm hot}^{(p)} + {\rm T}(n_{\rm RE,pri})$)')
    ax.plot(t[1:],Nre_ava+Ntp_ava,label=r'avalanche ($\Gamma_{\rm Ava}n_{\rm RE} + {\rm T}(n_{\rm RE,ava}$)')
    ax.plot(t[1:],Nre_pri+Nre_ava+Ntp,label=r'total ($\Phi_{\rm hot}^{(p)} + \Gamma_{\rm Ava}n_{\rm RE} + {\rm T}(n_{\rm RE})$)')
    #ax.plot(t[1:],-Ntp,label=r'-total transport (${\rm L_r}$)')
    #ax.plot(t[1:],-Ntp_pri-Ntp_ava,label=r'-total transport (${\rm L_r}\frac{n_{\rm RE,ava}}{n_{\rm RE}} + {\rm L_r}\frac{n_{\rm RE,ava}}{n_{\rm RE}}$)')
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("$N_{RE}$ (#)")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
"""
def survivingSeedFraction(outfile):
    unpert = np.genfromtxt('output/SSF/no_perturbation.txt')
    
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
    #print(f"SSF1 = {SSF1*100:1.2f}%")
    print(f"SSF = {SSF2*1000:1.2f}‰")
    print(f"SSF = {SSF4*1000:1.2f}‰")
    #np.savetxt('output/SSF/no_perturbation.txt',np.array([t[1:],Nre[1:],Nre_pri+Ntp,Nre_ava]).T)
    
    if True:
        fig, ax = plt.subplots(1,figsize=(5,3))
        ax.plot(unpert[:,0],unpert[:,1],c='C1',ls='--',label='Total w/o transport')
        ax.plot(unpert[:,0],unpert[:,2],c='C0',ls='--',label='Seed w/o transport')
        #ax.plot(unpert[:,0],unpert[:,3],c='C1',ls='--')
        ax.plot(t[:],Nre[:],c='C1',label=r'Total w/ transport')
        #ax.plot(t[1:],Nre_pri+Ntp_pri,label=r'seed ($\Phi_{\rm hot}^{(p)} + {\rm T}(n_{\rm RE})$)')
        #ax.plot(t[1:],Nre_pri+Ntp_pri,label=r'seed ($\Phi_{\rm hot}^{(p)} + {\mathrm{T}}(n_{\rm RE}-n_{\rm RE,ava})$)')
        #ax.plot(t[1:],Nre_pri+Ntp,label=r'seed ($\Phi_{\rm hot}^{(p)} + {\mathrm{T}}(n_{\rm RE})$)')
    #    ax.plot(t[1:],Nre_ava+Ntp_ava,label=r'avalanche ($\Gamma_{\rm Ava}n_{\rm RE} + {\rm T}(n_{\rm RE,ava}$)')
        #ax.plot(t[1:],Nre_ava,label=r'avalanche ($\Gamma_{\rm Ava}n_{\rm RE}$)')
        #ax.plot(t[1:],Nre_pri+Nre_ava+Ntp,label=r'total ($\Phi_{\rm hot}^{(p)} + \Gamma_{\rm Ava}n_{\rm RE} + {\rm T}(n_{\rm RE})$)')
        #ax.plot(t[1:],-Ntp,label=r'-total transport (${\rm L_r}$)')
        #ax.plot(t[1:],-Ntp_pri-Ntp_ava,label=r'-total transport (${\rm L_r}\frac{n_{\rm RE}-n_{\rm RE,ava}}{n_{\rm RE}} + {\rm L_r}\frac{n_{\rm RE,ava}}{n_{\rm RE}}$)')
        ax.set_xlabel("$t$ (s)")
        ax.set_ylabel("$N_{RE}$ (#)")
        ax.legend()
        #ax.set_xlim(0,90e-6)
        #ax.set_ylim(0,1e14)
        plt.tight_layout()
        plt.savefig('SSF_method.pdf')
        plt.show()

    return SSF1,SSF2,SSF3,SSF4,SSF5,SREF,avatco,Ipf
#"""
"""
def survivingSeedFraction(outfile,setfile):
    unpert = np.genfromtxt('output/SSF/no_perturbation.txt')
    
    do = DREAMOutput(outfile)
    ds = DREAMSettings(setfile)
    t  = do.grid.t[:]
    dt = np.diff(t)
    Ar  = ds.eqsys.n_re.transport.ar[:]
    Drr = ds.eqsys.n_re.transport.drr[:]
    transport_r = ds.eqsys.n_re.transport.todict()['drr']['r']
    transport_t = ds.eqsys.n_re.transport.todict()['drr']['t']
    
    # Total number of RE generated
    nre = do.eqsys.n_re[:]
    Nre = do.grid.integrate(nre)
    
    # Primary RE generated
    dnredt_prim = -do.other.fluid.gammaFhot[:]#/2.72515
    nre_prim = np.cumsum(dnredt_prim*dt[:,np.newaxis],axis=0)
    Nre_prim = do.grid.integrate(nre_prim)
    
    # Primary RE transported out of plasma edge
    Aa = interp1d(transport_t,Ar[:,-1],axis=0,fill_value="extrapolate")(t[1:])
    Da = interp1d(transport_t,Drr[:,-1],axis=0,fill_value="extrapolate")(t[1:])
    # FVM derivative at r=a with boundary condition nre(r>a)=0
    dnre_primdra = (0-nre_prim[:,-1])/do.grid.dr[-1]
    Ntp_prim = np.cumsum(do.grid.VpVol[-1]*(Aa*nre_prim[:,-1]/2+Da*dnre_primdra)*dt)#*1.6982647501506838

    dnredra = (0-nre[1:,-1])/do.grid.dr[-1]
    Ntp = np.cumsum(do.grid.VpVol[-1]*(Aa*nre[1:,-1]/2+Da*dnredra)*dt)
    
    Ntprl = np.cumsum(-do.other.scalar.radialloss_n_re[:].flatten()*dt)#*1.65
            
    # Avalanche generated RE for reference to see how good approx is
    dnredt_ava = do.other.fluid.GammaAva[:]*nre[:-1]
    nre_ava  = np.cumsum(dnredt_ava*dt[:,np.newaxis],axis=0)
    Nre_ava = do.grid.integrate(nre_ava)
    
    # SSF
    surviving_seed = Nre_prim[-1]
    unpert_seed = unpert[-1,1]
    print(f"SSF = {surviving_seed/unpert_seed*100:1.2f}%")

    fig, ax = plt.subplots(1)
#    ax.plot(unpert[:,0],unpert[:,3],c='C0',ls='--')
    #ax.plot(unpert[:,0],unpert[:,1],c='C1',ls='--')
    #ax.plot(unpert[:,0],unpert[:,2],c='C2',ls='--')
#    ax.plot(t[:],Nre[:],label=r'total ($n_{\rm RE}$)')
#    ax.plot(t[1:],Nre_prim,label=r'seed ($\Phi_{\rm hot}^{(p)}$)')
#    ax.plot(t[1:],Nre_ava,label=r'avalanche ($\Gamma_{\rm Ava}n_{\rm RE}$)')
#    ax.plot(t[1:],Ntp_prim,label='seed trnsp (operator)')
#    ax.plot(t[1:],Ntp,label='total trnsp (operator)')
#    ax.plot(t[1:],Ntprl,label='total trnsp (radialloss)')
    ax.plot(t[:],Nre[:],label=r'total ($n_{\rm RE}$)')
    ax.plot(t[1:],Nre_prim+Nre_ava+Ntp,label=r'seed ($\Phi_{\rm hot}^{(p)}$) + ava ($\Gamma_{\rm Ava}n_{\rm RE}$) + total trnsp (operator)')
    ax.plot(t[1:],Nre_prim+Nre_ava+Ntprl,label=r'seed ($\Phi_{\rm hot}^{(p)}$) + ava ($\Gamma_{\rm Ava}n_{\rm RE}$) + total trnsp (radialloss)')
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("$N_{RE}$ (#)")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
""""""
def survivingSeedFraction(filename,tco):
    unpert_Nre = np.genfromtxt('output/SSF/no_perturbation_Nre.txt') # Nre without magnetic perturbation
    unpert_SEED = np.genfromtxt('output/SSF/no_perturbation_seed.txt')
    
    do = DREAMOutput(filename)
    t  = do.grid.t[:]
    dt = np.diff(t)
    tco = float(tco)  

    # Total number of RE generated, which approximates the seed at short times    
    dnredt = do.other.fluid.runawayRate[:]
    nre = np.cumsum(dnredt*dt[:,np.newaxis],axis=0)
    Nre = do.grid.integrate(nre)
    
    # Avalanche generated RE for reference to see how good approx is
    dnredt_ava = do.other.fluid.GammaAva[:]*do.eqsys.n_re[1:]
    nre_ava  = np.cumsum(dnredt_ava*dt[:,np.newaxis],axis=0)
    Nre_ava = do.grid.integrate(nre_ava)
    
    # SSF
    surviving_seed = Nre[t[1:]<=tco][-1]
    unpert_seed = unpert_Nre[-1,1]
    print(f'Nre(tco,trans)/Nre(plateau,no trans)  = {surviving_seed/unpert_seed*100:1.2f}%')
    print(f'Nre(tco,trans)/seed(plateau,no trans) = {surviving_seed/unpert_SEED[-1,1]*100:1.2f}%')
    print(f'(Nre-Nava)(plateau,trans)/Nre(plateau,no trans)  = {(Nre-Nre_ava)[-1]/unpert_seed*100:1.2f}%')
    print(f'(Nre-Nava)(platuea,trans)/seed(plateau,no trans) = {(Nre-Nre_ava)[-1]/unpert_SEED[-1,1]*100:1.2f}%')
    
    fig, ax = plt.subplots(1)
    ax.plot(unpert_Nre[:,0],unpert_Nre[:,1],label='$N_{RE}$, no transport',c='C0')
    ax.plot(unpert_SEED[:,0],unpert_SEED[:,1],label='seed, no transport',c='C0',ls='--')
    ax.plot(t[1:],Nre,label='$N_{RE}$, transport',c='C1')
    ax.plot(t[1:],Nre-Nre_ava,label="'seed', transport",c='C1',ls='--')
    #ax.plot(t[1:],Nre_ava,label='$N_{RE,avalanche}$',c='C2')
    ax.axvline(tco,c='grey',ls=':')
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("$N_{RE}$ (#)")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
"""
#"""
def dBB_sensitivity(filename):
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
    axA.set_xlabel(r"$\left(\delta B/B\right)_0 \cdot 10^{3}$ (#)")
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

    
def radial_density_influence(filename):
    do = DREAMOutput(filename)
    cmap = plt.get_cmap('GeriMap')
    times = [50,55,60,65,70,75,80,-1]
    mask = lambda a,b: a if ti not in [75,-1] else b
    for i,ti in enumerate(times):
        plt.plot(do.grid.r[:]/0.5, do.eqsys.n_re[ti,:]/do.eqsys.n_re[ti,:].max(), c=cmap(i/len(times)),
                 label=fr"$t=${do.grid.t[ti]*1e6:1.0f} $\rm\mu$s",alpha=mask(0.3,1),ls=mask('--','-'),lw=mask(1.5,2.5))
    plt.ylabel(r"$n_{\rm RE}/\max{(n_{\rm RE})}$")
    plt.xlabel("$r/a$")
    plt.legend()
    plt.show()
    
def radial_density_influence_FULL(filename):
    path = "output/scans_dBB-profile/Advection_dominated/output_idBBr"
    dos = [DREAMOutput(path+"0.h5"),DREAMOutput(path+"1_core-raised.h5"),DREAMOutput(path+"0_A3D3.h5"),DREAMOutput(path+"1_core-raised_A3D3.h5")]
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
    
def SSF_scan_plot(filename):
    data = np.genfromtxt(filename,delimiter=', ',skip_header=1)
    tco = data[:,0]
    A0  = data[:,1]
    SSF1 = data[:,2] # SSF -full tp (lower)
    SSF2 = data[:,3] # SSF nRE at tco
    SSF3 = data[:,4] # SSF nRE at 100us
    SSF4 = data[:,5] # SSF nRE at tco+pri.gen.rest. (higher if ava.gen.>ava.tp. which is always?)
    SSF5 = data[:,6] # SSF -partial tp (higher)
    SREF = data[:,7] # SREF nRE at final
    AVAg = data[:,8] # generated ava Nre at tco
    Ipf  = data[:,9] # final current error

    fig, axA = plt.subplots(1,figsize=(7,4))
    axt = axA.twiny()
    
    #axA.semilogx(A0*1e3,((Ipf-200e3)/Ipf)*5e2+5.5,'-o',c='C2',ls=':',label="")
    axA.semilogx(A0*1e3,AVAg*1e2+3.67,'-o',c='grey',ls='--',label=r"$N_{\mathrm{RE,ava}}(t_{\mathrm{cutoff}})$ (a.u.)")
    axA.semilogx(A0*1e3,SSF4*1e4,'-o',c='C0',label="SSF")
    #axA.semilogx(A0*1e3,SSF3,'-o')
    axt.semilogx(A0*1e3,SSF4*1e4,alpha=0,c='C0')
    
    axA.set_ylim(3.5,6)
    axA.legend(reverse=True)
    axA.minorticks_off()
    axt.minorticks_off()
    axA.set_xticks(A0*1e3,[f'{a:1.1f}' for a in A0*1e3])    
    axt.set_xticks(A0*1e3,[f'{t:1.0f}' for t in tco*1e6])
    axA.xaxis.grid(color='grey',ls=':',alpha=0.8)
    axA.set_xlabel(r"$\left(\delta B/B\right)_0\cdot 10^{3}$ (#)")
    axt.set_xlabel(r"$t_{\rm cutoff}$ ($\rm \mu$s)")
    axA.set_ylabel("Surviving seed fraction (10$^{-4}$)")
    plt.tight_layout()
    plt.savefig('SSF.pdf')
    plt.show()
    
def SSF_scan_plot2():
    path = 'output/scans_dBB-profile/constant_radial_b250_sensitivity/'
    files = ['tcoscan1_output_0.h5', 'tcoscan1_output_1.h5', 'tcoscan1_output_2.h5', 'tcoscanX_outputBSX_0.h5', 'tcoscan_output_1.h5', 'tcoscan_output_2.h5', 'tcoscanX_outputBSX_1.h5', 'tcoscan_output_3.h5', 'tcoscan_output_4.h5', 'outputBX.h5', 'tcoscan_output_5.h5']
    dos = [DREAMOutput(path+f) for f in files]
    
    for i,do in enumerate(dos):
        #plt.plot(do.grid.t[:],do.grid.integrate(do.eqsys.n_re[:])*do.grid.R0[0],label=fr'{35+i*5:1.0f} $\rm\mu$s',ls=['-','--'][i>=8])
        plt.plot(do.grid.t[1:],do.grid.integrate(do.eqsys.E_field[1:]/do.other.fluid.Eceff[:])*do.grid.R0[0],label=fr'{35+i*5:1.0f} $\rm\mu$s',ls=['-','--'][i>=8])
    plt.legend()
    plt.xlabel("$t$ (s)")
    plt.ylabel(r"$N_{\rm RE}$ (#)")
    plt.show()

"""    
def SSF_scan_plot2():
    path = 'output/scans_dBB-profile/constant_radial_b250_sensitivity/'
    files = ['tcoscan1_output_0.h5', 'tcoscan_output_5.h5']
    dos = [DREAMOutput(path+f) for f in files]
    
    for i,do in enumerate(dos):
        plt.plot(do.grid.t[:],do.eqsys.n_re[:],ls=['-','--'][i])
    plt.xlabel("$t$ (s)")
    plt.ylabel(r"$n_{\rm RE}$ (#)")
    plt.show()
"""

def temp_dBB_plot(filenames):
    fig, axs = plt.subplots(1,2,figsize=(9.3,4),gridspec_kw={'width_ratios':[1,1.2]})
    
    for filename, ax in zip(filenames,axs):
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
        ax.set_ylabel("$t_{cut off}$ (µs)")
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
    
def temp_SSF_plot(outfile,SFF_file):
    unpert = np.genfromtxt('output/SSF/no_perturbation.txt')
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


## CALL SELECTED PLOT FUNCTION
def MAIN(argv):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#000000', '#0572f0', '#ff8103', '#13bd2a', '#f70227', 
                                                        '#f7cf19','#c719f7','#27d6d0'])
    if argv[0] == 'scan':
        fig, ax = plt.subplots(1)
        for filename in argv[1:]:
            parameterScanPlot(fig,ax,filename)
        #for p in argv[1:]:
            #__plot_A0__(fig,ax,argv[0],float(p))
            #__plot_tco__(fig,ax,argv[0],float(p))
        plt.show()
    if argv[0] == 'detect': detective()
    elif argv[0] == 'dBB': dBB_parameterPlot2D(argv[1])
    elif argv[0] == 'sens': dBB_sensitivity(argv[1])
    elif argv[0] == 'SSF': survivingSeedFraction(*argv[1:])
    elif argv[0] == 'SSFplot': SSF_scan_plot(argv[1])
    elif argv[0] == 'SSFplot2': SSF_scan_plot2()
    elif argv[0] == 'dens': radial_density_influence_FULL(argv[1])
    elif argv[0] == 'temp': temp_dBB_plot(argv[1:])
    else: print("The first commandline argument should be of:"
                "scan, detect, dBB, sens, SSF or dens")
    
if __name__ == '__main__':
    sys.exit(MAIN(sys.argv[1:]))
