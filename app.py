import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from models import *

@st.cache_data(show_spinner=False)
def cached_gmm(r,dt,model):
    return fit_gmm(np.asarray(r),dt,model)

from design import setup, page_heading
setup('Kaoutar ETTAFSSAOUI')
options=['Vue d’ensemble', 'Bootstrap', 'Interpolation & lissage', 'Nelson–Siegel / Svensson', 'Vasicek / CIR', 'Estimation GMM', 'Laboratoire interactif']
page=st.radio('Navigation', options, horizontal=True, key='navigation', label_visibility='collapsed', format_func=lambda x: dict(zip(options,['Vue générale','Bootstrap','Splines','Nelson–Siegel','Vasicek / CIR','GMM','Explorer']))[x])
if page=='Laboratoire interactif':
    from explorer import render
    render()
    st.stop()
page_heading(options.index(page))


def chart(xs,series,title,ylabel='Taux (%)'):
    from design import plot
    plot(xs,series,title,ylabel)


def export(df,name):
    st.download_button('Télécharger les résultats CSV',df.to_csv(index=False).encode('utf-8-sig'),name,'text/csv',key=name)


def bonds():
    with st.expander('Données obligataires et conventions',expanded=False):
        st.write('Coupons annuels, nominal remboursé au pair, évaluation à une date de coupon. Prix plein coupon. Maturités complètes 1, 2, …, N. Le coupon est un montant, pas un taux.')
        up=st.file_uploader('Importer un CSV : maturity, coupon, nominal, price',type='csv',key=page+'_csv')
        st.caption('CSV séparé par une virgule ou un point-virgule ; point comme séparateur décimal.')
        demo=demo_bonds();st.download_button('Télécharger le modèle CSV',demo.to_csv(index=False),'obligations_exemple.csv',key=page+'_template')
    d=validate_bonds(pd.read_csv(up,sep=None,engine='python')) if up else demo_bonds()
    st.caption('Données importées — source à documenter.' if up else 'Démonstration synthétique · 20 obligations · nominal 100 DH · coupon annuel 5 DH. Aucune cotation réelle.')
    b=bootstrap(d)
    return d,b

if page==options[0]:
    from design import overview
    overview(options, demo_bonds, bootstrap, chart)

elif page in ['Bootstrap','Interpolation & lissage','Nelson–Siegel / Svensson']:
  try:
    d,b=bonds();t=d.maturity.to_numpy();grid=np.linspace(.01,t[-1],400);f=cashflows(d)
    if page=='Bootstrap':
        r=-np.log(b)/t;rebuilt=f@b
        cols=st.columns(3);cols[0].metric('Obligations',len(d));cols[1].metric('Taux à 1 an',f'{r[0]*100:.3f} %');cols[2].metric('Erreur maximale de prix',f'{np.max(abs(rebuilt-d.price)):.2e} DH')
        chart(t,{'Taux zéro-coupon':r*100},'Reconstruction par maturité croissante')
        out=d.assign(discount=b,rate_continuous=r,price_rebuilt=rebuilt,residual=rebuilt-d.price)
        st.dataframe(out,width='stretch');export(out,'bootstrap_resultats.csv')
        st.latex(r'B(0,n)=\frac{P_n-C_n\sum_{k=1}^{n-1}B(0,k)}{C_n+N_n},\quad R(0,n)=-\frac{\ln B(0,n)}n')
        if np.any(np.diff(b)>0):st.warning('Certains facteurs augmentent : examiner les taux forward et les données ; ceci peut être compatible avec des taux négatifs.')
    elif page=='Interpolation & lissage':
        method=st.selectbox('Méthode',['Linéaire sur les taux','Spline cubique naturelle','Spline exponentielle cubique'])
        knots=[];coef=None
        if method=='Linéaire sur les taux':
            fn=lambda x:np.exp(-np.asarray(x)*np.interp(x,t,-np.log(b)/t))
            st.caption('Entre 0 et 1 an : taux maintenu au taux à 1 an. Interpolation ensuite entre les points connus.')
        elif method=='Spline cubique naturelle':
            fn,coef=fit_spline(d,'cubique',[])
            st.caption('Interpolation exacte des facteurs avec B(0)=1 et dérivées secondes nulles aux deux extrémités.')
        else:
            c=st.columns(3)
            raw=c[0].text_input('Raccords (années, séparés par virgules)','5,10' if len(d)>10 else '1')
            u=c[1].number_input('Échelle u > 0',.01,2.,.15,.01)
            pen=c[2].number_input('Pénalité sur ∫ B″²',0.,100000.,0.,10.)
            knots=sorted(set(float(v.strip()) for v in raw.split(',') if v.strip()))
            if any(k<=0 or k>=t[-1] for k in knots):raise ValueError('Les raccords doivent être strictement à l’intérieur du domaine.')
            fn,coef=fit_spline(d,'expo',knots,u,pen)
            st.caption('Ajustement des prix sur une base exponentielle C² ; u est fixé par le curseur. Positivité contrôlée sur une grille, sans garantie hors grille ni extrapolation.')
        bg=fn(grid)
        if np.any(bg<=0):raise ValueError('La courbe produit un facteur non positif : modifier la méthode ou les paramètres.')
        rates=-np.log(bg)/grid;re=f@fn(t);err=re-d.price.to_numpy()
        st.metric('RMSE des prix (DH)',f'{np.sqrt(np.mean(err**2)):.6f}')
        chart(grid,{method:rates*100},'Courbe interpolée / ajustée')
        out=pd.DataFrame({'maturity':grid,'discount':bg,'rate_continuous':rates});export(out,'splines_courbe.csv')
        st.dataframe(pd.DataFrame({'maturity':t,'price':d.price,'reconstructed':re,'residual':err}),width='stretch')
        if coef is not None:st.write('Coefficients de la base normalisée',coef)
    else:
        st.write('Ajustement sur les prix, avec plusieurs initialisations. Les paramètres d’échelle sont strictement positifs.')
        @st.cache_data(show_spinner=False)
        def fit_all(data):return [(sv,fit_ns(data,sv)[0]) for sv in [False,True]]
        with st.spinner('Estimation des deux modèles…'): fits=fit_all(d)
        series={};comparison=[];param=[];out=pd.DataFrame({'maturity':grid})
        for sv,p in fits:
            name='Svensson' if sv else 'Nelson–Siegel';rates=ns_rate(grid,p,sv);series[name]=rates*100;out[name]=rates
            err=f@np.exp(-t*ns_rate(t,p,sv))-d.price.to_numpy()
            comparison.append({'Modèle':name,'RMSE prix (DH)':np.sqrt(np.mean(err**2)),'MAE prix (DH)':np.mean(abs(err))})
            names=['beta0','beta1','beta2','lambda1']+(['beta3','lambda2'] if sv else [])
            param.extend({'Modèle':name,'Paramètre':k,'Valeur':v} for k,v in zip(names,p))
        chart(grid,series,'Comparaison des modèles paramétriques')
        st.dataframe(pd.DataFrame(comparison),width='stretch');st.dataframe(pd.DataFrame(param),width='stretch')
        export(out,'nelson_siegel_courbes.csv');export(pd.DataFrame(param),'parametres_ns_svensson.csv')
        st.caption('Résultats conditionnels aux bornes d’optimisation et aux initialisations ; pas de garantie de minimum global. Une RMSE plus faible ne prouve pas une meilleure extrapolation.')
  except (ValueError,KeyError,TypeError) as e:st.error(str(e))

elif page=='Vasicek / CIR':
    model=st.selectbox('Modèle',['Vasicek','CIR'])
    st.info('Paramètres ci-dessous sous Q pour la valorisation. Les trajectoires simulées utilisent également ces paramètres Q ; ce ne sont pas des prévisions historiques.')
    c=st.columns(4)
    a=c[0].number_input('Vitesse a',.01,10.,.5,.05)
    b=c[1].number_input('Niveau long terme (%)',.01 if model=='CIR' else -10.,30.,4.,.1)/100
    s=c[2].number_input('Coefficient de diffusion',.0001,1.,.08 if model=='CIR' else .01,.005,format='%.4f')
    r0=c[3].number_input('Taux initial (%)',0. if model=='CIR' else -10.,30.,3.,.1)/100
    horizon=st.slider('Horizon (années)',1,30,10);seed=st.number_input('Graine aléatoire',0,100000,42)
    paths=simulate(a,b,s,r0,1/12,horizon*12,20,model,int(seed));ts=np.arange(len(paths))/12
    chart(ts,{f'Trajectoire {j+1}':paths[:,j]*100 for j in range(8)},'Trajectoires simulées — transitions exactes')
    maturities=np.linspace(.01,30,300);prices=bond_price(maturities,r0,a,b,s,model);rates=-np.log(prices)/maturities
    chart(maturities,{'Taux zéro-coupon':rates*100},'Courbe de valorisation sous Q')
    if model=='CIR':
        ratio=2*a*b/s**2;st.metric('Rapport de Feller : 2ab / σ²',f'{ratio:.3f}')
        if ratio<1:st.warning('Zéro est accessible. Le schéma de transition exact conserve la non-négativité.')
    else:st.caption('Vasicek autorise des taux négatifs.')
    export(pd.DataFrame({'maturity':maturities,'discount':prices,'rate_continuous':rates}),'modele_courbe.csv')
    export(pd.DataFrame(paths,index=ts).rename_axis('time').reset_index(),'trajectoires.csv')

else:
  st.info('GMM historique : moments conditionnels exacts, instruments 1 et taux standardisé, deux étapes et pondération HAC régularisée. Ces paramètres ne sont pas automatiquement des paramètres risque neutre.')
  model=st.selectbox('Modèle à estimer',['Vasicek','CIR'])
  source=st.radio('Série',['Démonstration synthétique','Importer CSV'],horizontal=True)
  true=np.array([.7,.04,.012 if model=='Vasicek' else .10]);dt=1/12
  try:
    if source=='Démonstration synthétique':
        seed=st.number_input('Graine de la série',0,100000,42);N=st.slider('Nombre de transitions mensuelles',240,2400,1200,120)
        r=simulate(*true,.03,dt,N,1,model,int(seed))[:,0]
        st.caption('Série synthétique de validation, sans observations réelles. Une réalisation finie ne permet pas de retrouver exactement les paramètres.')
    else:
        st.write('CSV : colonne rate (taux), éventuellement date. Observations régulièrement espacées et triées chronologiquement ; au moins 100 lignes.')
        up=st.file_uploader('Série historique CSV',type='csv');unit=st.selectbox('Unité',['Décimal (0.04)','Pourcentage (4)'])
        dt=st.number_input('Pas Δ en années',.0001,2.,1/12,format='%.6f')
        if up is None:st.stop()
        data=pd.read_csv(up,sep=None,engine='python')
        if 'date' in data:
            data['date']=pd.to_datetime(data.date,errors='raise');data=data.sort_values('date')
            if data.date.duplicated().any():raise ValueError('Dates dupliquées.')
        r=pd.to_numeric(data['rate'],errors='raise').to_numpy();r=r/(100 if unit.startswith('Pourcentage') else 1)
    chart(np.arange(len(r))*dt,{'Taux observé':r*100},'Série utilisée pour la GMM')
    export(pd.DataFrame({'rate':r}),'serie_gmm.csv')
    import hashlib
    signature=(model, float(dt), hashlib.sha256(np.asarray(r).tobytes()).hexdigest())
    if st.button('Estimer les paramètres par GMM',type='primary'):
        st.session_state['gmm_signature']=signature
    if st.session_state.get('gmm_signature')==signature:
        with st.spinner('Estimation en deux étapes…'):p,obj,mom,bound=cached_gmm(tuple(r),dt,model)
        result=pd.DataFrame({'Paramètre':['a','b','sigma'],'Estimation':p})
        if source=='Démonstration synthétique':result['Vraie valeur']=true
        st.dataframe(result,width='stretch');st.metric('Critère GMM final',f'{obj:.6g}')
        st.write('Moyennes des quatre moments normalisés',mom)
        if bound:st.warning('Un paramètre atteint une borne : estimation potentiellement instable, à ne pas interpréter sans diagnostic.')
        if model=='CIR':st.write('Rapport de Feller estimé :',float(2*p[0]*p[1]/p[2]**2))
        export(result,'gmm_parametres.csv')
        st.caption('Pas d’intervalles de confiance dans cette version. Examiner aussi la sensibilité à la taille de l’échantillon et aux hypothèses du modèle.')
  except (ValueError,KeyError,TypeError) as e:st.error(str(e))
