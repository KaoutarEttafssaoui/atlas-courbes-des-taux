"""Calculs du projet. Temps en années, taux décimaux, prix plein coupon."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, minimize


def demo_bonds(n=20):
    t=np.arange(1,n+1,dtype=float)
    r=.025+.012*(-np.expm1(-t/5))+.006*(t/3)*np.exp(1-t/3)
    b=np.exp(-t*r)
    p=np.array([5*b[:j].sum()+105*b[j] for j in range(n)])
    return pd.DataFrame(dict(maturity=t,coupon=np.full(n,5.),nominal=np.full(n,100.),price=p))


def validate_bonds(df):
    cols=['maturity','coupon','nominal','price']
    if not set(cols)<=set(df.columns): raise ValueError('Colonnes requises : '+', '.join(cols))
    d=df[cols].apply(pd.to_numeric,errors='raise').sort_values('maturity').reset_index(drop=True)
    if not np.isfinite(d.values).all(): raise ValueError('Valeurs manquantes ou non finies.')
    if len(d)<2 or len(d)>60: raise ValueError('Utiliser 2 à 60 obligations.')
    if not np.allclose(d.maturity,np.arange(1,len(d)+1)): raise ValueError('Cette version exige les maturités annuelles complètes 1, 2, …, N, sans doublon.')
    if (d.coupon<0).any() or (d.nominal<=0).any() or (d.price<=0).any(): raise ValueError('Prix et nominal strictement positifs ; coupons positifs ou nuls.')
    return d


def cashflows(d):
    n=len(d); f=np.zeros((n,n))
    for j,row in d.iterrows():
        f[j,:j+1]=row.coupon; f[j,j]+=row.nominal
    return f


def bootstrap(d):
    b=np.zeros(len(d))
    for j,row in d.iterrows(): b[j]=(row.price-row.coupon*b[:j].sum())/(row.nominal+row.coupon)
    if np.any(b<=0): raise ValueError('Bootstrap impossible : facteur non positif. Vérifier prix et conventions.')
    return b


def loading(x):
    x=np.asarray(x,float)
    return -np.expm1(-x)/np.maximum(x,1e-12)


def ns_rate(t,p,sv=False):
    t=np.asarray(t,float); x=t/p[3]; l=np.where(t==0,1.,loading(x))
    r=p[0]+p[1]*l+p[2]*(l-np.exp(-x))
    if sv:
        z=t/p[5]; l2=np.where(t==0,1.,loading(z)); r=r+p[4]*(l2-np.exp(-z))
    return r


def fit_ns(d,sv=False):
    if len(d)<(8 if sv else 5): raise ValueError('Au moins 8 obligations pour Svensson, 5 pour Nelson–Siegel.')
    t=d.maturity.to_numpy(); f=cashflows(d); prices=d.price.to_numpy()
    def residual(p): return f@np.exp(-t*ns_rate(t,p,sv))-prices
    low=[-.15,-.3,-.3,.1]+([-.3,.1] if sv else [])
    high=[.3,.3,.3,40]+([.3,40] if sv else [])
    sols=[]
    for scale in [1.,3.,8.]:
        p=[.035,-.015,.02,scale]+([-.01,scale*2] if sv else [])
        res=least_squares(residual,p,bounds=(low,high),max_nfev=2500)
        if res.success and np.isfinite(res.cost): sols.append(res)
    if not sols: raise ValueError('Optimisation non convergée.')
    res=min(sols,key=lambda x:x.cost)
    return res.x,lambda x:np.exp(-np.asarray(x)*ns_rate(x,res.x,sv)),res.message


def exp_basis(t,knots,u):
    t=np.asarray(t,float); z=np.exp(-u*t)
    return np.column_stack([z-1,z*z-1,z*z*z-1]+[np.maximum(np.exp(-u*k)-z,0)**3 for k in knots])


def fit_spline(d,kind,knots,u=.15,penalty=0.):
    t=d.maturity.to_numpy(); b=bootstrap(d)
    if kind=='cubique':
        return CubicSpline(np.r_[0,t],np.r_[1,b],bc_type='natural',extrapolate=False),None
    knots=np.asarray(knots,float)
    if len(t)<3+len(knots): raise ValueError('Nombre de coefficients supérieur au nombre de prix disponibles.')
    grid=np.linspace(0,t[-1],401); X=exp_basis(t,knots,u); f=cashflows(d)
    G=exp_basis(grid,knots,u); h=grid[1]-grid[0]; D=np.diff(G,n=2,axis=0)/h**2
    target=d.price.to_numpy()-f@np.ones(len(t)); M=f@X
    if penalty>0:
        M=np.vstack([M,np.sqrt(penalty*h)*D]); target=np.r_[target,np.zeros(len(D))]
    coef=np.linalg.lstsq(M,target,rcond=None)[0]
    # C2 exponential basis, B(0)=1; enforce positivity on a dense grid.
    if np.min(1+G@coef)<1e-8:
        objective=lambda c:np.sum((M@c-target)**2)
        opt=minimize(objective,coef,method='SLSQP',constraints={'type':'ineq','fun':lambda c:1+G@c-1e-8},options={'maxiter':1500,'ftol':1e-10})
        if not opt.success: raise ValueError('Ajustement sous contrainte non convergé : '+opt.message)
        coef=opt.x
    return lambda x:1+exp_basis(x,knots,u)@coef,coef


def transition(r,a,b,s,dt,model):
    q=np.exp(-a*dt); z=-np.expm1(-a*dt); mu=b+(np.asarray(r)-b)*q
    v=s*s*z*(1+q)/(2*a) if model=='Vasicek' else s*s*np.asarray(r)*q*z/a+b*s*s*z*z/(2*a)
    return mu,v


def simulate(a,b,s,r0,dt,n,paths,model,seed=42):
    rng=np.random.default_rng(seed); out=np.zeros((n+1,paths));out[0]=r0
    for i in range(n):
        if model=='Vasicek':
            mu,v=transition(out[i],a,b,s,dt,model);out[i+1]=mu+np.sqrt(v)*rng.normal(size=paths)
        else:
            q=np.exp(-a*dt); z=-np.expm1(-a*dt); c=s*s*z/(4*a);df=4*a*b/(s*s)
            nc=4*a*q*out[i]/(s*s*z);out[i+1]=c*rng.noncentral_chisquare(df,nc,size=paths)
    return out


def bond_price(t,r,a,b,s,model):
    t=np.asarray(t,float)
    if model=='Vasicek':
        A=-np.expm1(-a*t)/a; C=(b-s*s/(2*a*a))*(A-t)-s*s*A*A/(4*a)
        return np.exp(-A*r+C)
    ell=np.sqrt(a*a+2*s*s); z=-np.expm1(-ell*t)
    den=(a+ell)*z+2*ell*np.exp(-ell*t)
    A=2*z/den
    logpsi=np.log(2*ell)+(a-ell)*t/2-np.log(den)
    return np.exp((2*a*b/(s*s))*logpsi-A*r)


def fit_gmm(r,dt,model):
    r=np.asarray(r,float)
    if len(r)<100 or not np.isfinite(r).all(): raise ValueError('Au moins 100 taux finis sont nécessaires.')
    if np.std(r)<1e-7: raise ValueError('Série presque constante : paramètres non identifiables.')
    if model=='CIR' and np.any(r<0): raise ValueError('Le CIR standard ne prend pas en charge les taux négatifs.')
    x,y=r[:-1],r[1:]; ins=(x-x.mean())/x.std(); scale=max(np.std(np.diff(r)),1e-6)
    def moments(p):
        mu,v=transition(x,*p,dt,model); e=y-mu; h=e*e-v
        return np.column_stack([e/scale,ins*e/scale,h/scale**2,ins*h/scale**2])
    bounds=[(.001,20),((.00001 if model=='CIR' else -.2),.5),(.00001,2)]
    def optimize(W):
        def obj(p):
            g=moments(p).mean(axis=0);return g@W@g
        sols=[]
        sig=scale/np.sqrt(dt)
        if model=='CIR':sig/=np.sqrt(max(x.mean(),.001))
        for a in [.1,1.,3.]:
            p=[a,np.clip(x.mean(),bounds[1][0]+1e-6,.49),np.clip(sig,.0001,1.9)]
            res=minimize(obj,p,method='L-BFGS-B',bounds=bounds,options={'maxiter':1500,'ftol':1e-12})
            if res.success and np.isfinite(res.fun):sols.append(res)
        if not sols:raise ValueError('GMM : aucune initialisation convergée.')
        return min(sols,key=lambda o:o.fun)
    first=optimize(np.eye(4)); g=moments(first.x); g-=g.mean(axis=0); n=len(g)
    S=g.T@g/n; lag=min(int(4*(n/100)**(2/9)),n-1)
    for k in range(1,lag+1):
        cov=g[k:].T@g[:-k]/n;S+=(1-k/(lag+1))*(cov+cov.T)
    W=np.linalg.pinv(S+1e-8*np.eye(4));res=optimize(W)
    boundary=any(abs(res.x[i]-lo)<1e-4 or abs(res.x[i]-hi)<1e-4 for i,(lo,hi) in enumerate(bounds))
    return res.x, float(res.fun), moments(res.x).mean(axis=0),boundary
