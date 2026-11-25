"""Interactive model exploration; all scenarios use risk-neutral parameters."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from models import simulate, bond_price

TEAL='#197b73'

def animated_paths(times,paths):
    colors=['#197b73','#c38b40','#637ea5','#ae6683','#7e8e62','#7870a2']
    count=min(paths.shape[1],6)
    frame_ids=np.unique(np.linspace(0,len(times)-1,min(len(times),61),dtype=int))
    def traces(k):
        return [go.Scatter(x=times[:k+1],y=paths[:k+1,j]*100,mode='lines',name=f'Trajectoire {j+1}',line=dict(width=2,color=colors[j])) for j in range(count)]
    frames=[go.Frame(data=traces(k),name=str(k),traces=list(range(count))) for k in frame_ids]
    fig=go.Figure(data=traces(0),frames=frames)
    low=float(paths[:,:count].min()*100);high=float(paths[:,:count].max()*100)
    pad=max((high-low)*.12,.1)
    fig.update_layout(template='plotly_white',height=530,margin=dict(t=30,b=110,l=50,r=25),font=dict(color='#203b38'),xaxis=dict(title='Temps simulé (années)',range=[0,float(times[-1])],autorange=False),yaxis=dict(title='Taux court (%)',range=[low-pad,high+pad],autorange=False),legend=dict(orientation='h',y=1.12),hovermode='x unified',
        updatemenus=[dict(type='buttons',direction='left',x=0,y=-.18,buttons=[
            dict(label='▶ Lire / reprendre',method='animate',args=[None,dict(frame=dict(duration=100,redraw=False),transition=dict(duration=0),fromcurrent=True,mode='immediate')]),
            dict(label='❚❚ Pause',method='animate',args=[[None],dict(frame=dict(duration=0,redraw=False),mode='immediate')]),
            dict(label='↺ Recommencer',method='animate',args=[[str(k) for k in frame_ids],dict(frame=dict(duration=100,redraw=False),transition=dict(duration=0),mode='immediate',fromcurrent=False)])])],
        sliders=[dict(active=0,x=.02,len=.96,y=-.06,currentvalue=dict(prefix='Temps simulé : ',suffix=' ans'),steps=[dict(label=f'{times[k]:.1f}',method='animate',args=[[str(k)],dict(mode='immediate',frame=dict(duration=0,redraw=False),transition=dict(duration=0))]) for k in frame_ids])])
    return fig

def render():
    st.markdown('<div class="eyebrow">ATLAS / 07 — Laboratoire interactif</div>',unsafe_allow_html=True)
    st.title('Explorez la dynamique des taux.')
    st.write('Animez les trajectoires et observez l’effet des paramètres sur les taux.')
    st.caption('Scénarios simulés sous la probabilité risque neutre Q. Aucune cotation réelle ni prévision historique.')
    with st.container(border=True):
        cols=st.columns([1.3,1,1,1,1])
        model=cols[0].selectbox('Modèle',['Vasicek','CIR'],key='lab_model')
        a=cols[1].slider('Vitesse a',.05,3.,.7,.05,key='lab_a')
        b=cols[2].slider('Niveau long terme (%)',.1,12.,4.,.1,key='lab_b')/100
        s=cols[3].slider('Diffusion σ',.005,.25,.10,.005,key='lab_sigma_cir') if model=='CIR' else cols[3].slider('Diffusion σ',.001,.05,.012,.001,key='lab_sigma_vas')
        r0=cols[4].slider('Taux initial (%)',.1,12.,3.,.1,key='lab_r0')/100
        c1,c2=st.columns(2)
        horizon=c1.select_slider('Horizon simulé (années)',options=[2,5,10,15,20],value=5,key='lab_horizon')
        seed=c2.number_input('Graine du scénario',min_value=0,max_value=100000,value=42,key='lab_seed',help='Une graine fixe permet de comparer les réglages avec la même source aléatoire.')
    times=np.arange(horizon*12+1)/12
    tenors=np.linspace(.25,30,80)
    initial_curve=-np.log(bond_price(tenors,r0,a,b,s,model))/tenors*100
    cols=st.columns(3)
    cols[0].metric('Taux court initial',f'{r0*100:.2f} %')
    cols[1].metric('Taux zéro-coupon initial · 30 ans',f'{initial_curve[-1]:.2f} %')
    cols[2].metric('Modèle sous Q',model)
    if model=='CIR' and 2*a*b<s*s:
        st.warning('Condition de Feller non vérifiée : zéro est accessible. La simulation utilise une transition exacte et reste non négative.')
    if model=='Vasicek':st.caption('Le modèle de Vasicek autorise des taux négatifs.')
    animation,sensitivity=st.tabs(['Trajectoires animées','Sensibilité des courbes'])
    with animation:
        st.markdown('#### Regarder les trajectoires se construire')
        st.caption('Cliquez sur Lire. Le curseur temporel permet de revenir à un instant précis. L’animation révèle des trajectoires déjà simulées ; elle n’ajoute pas de nouvelles observations.')
        paths=simulate(a,b,s,r0,1/12,horizon*12,6,model,int(seed))
        fig=animated_paths(times,paths)
        st.plotly_chart(fig,width='stretch',key='animated_paths',config={'displaylogo':False})
        st.download_button('Exporter les trajectoires CSV',pd.DataFrame(paths,index=times).rename_axis('simulated_time').reset_index().to_csv(index=False).encode('utf-8-sig'),'trajectoires_animees.csv','text/csv')
    with sensitivity:
        st.markdown('#### Comparer trois scénarios de taux initial')
        shock=st.slider('Écart autour du taux initial (points de base)',5,100,25,5,key='lab_shock')/10000
        candidates=[r0-shock,r0,r0+shock]
        if model=='CIR' and candidates[0]<0:
            candidates[0]=0.;st.info('Le scénario bas est limité à zéro pour respecter le domaine du CIR.')
        fig=go.Figure();out=pd.DataFrame({'remaining_maturity':tenors})
        for label,r,color in zip(['Taux initial bas','Référence','Taux initial haut'],candidates,['#c38b40',TEAL,'#637ea5']):
            curve=-np.log(bond_price(tenors,r,a,b,s,model))/tenors*100
            fig.add_trace(go.Scatter(x=tenors,y=curve,name=f'{label} ({r*100:.2f} %)',line=dict(color=color,width=3)))
            out[label]=curve/100
        fig.update_layout(template='plotly_white',height=400,hovermode='x unified',xaxis_title='Maturité restante (années)',yaxis_title='Taux zéro-coupon (%)',legend=dict(orientation='h',y=-.2))
        st.plotly_chart(fig,width='stretch',key='sensitivity')
        st.caption('Comparaison instantanée sous Q : seul le taux court initial varie. a, b et σ restent identiques entre les trois courbes.')
        st.download_button('Exporter la comparaison CSV',out.to_csv(index=False).encode('utf-8-sig'),'sensibilite_taux.csv','text/csv')
