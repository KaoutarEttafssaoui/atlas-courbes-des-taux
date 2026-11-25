import html
import numpy as np
import streamlit as st
import plotly.graph_objects as go

COLORS=['#197b73','#c38b40','#637ea5','#ae6683','#7e8e62','#7870a2','#c97556','#4d919d']

def setup(author):
    st.set_page_config(page_title='ATLAS | Courbes des taux',page_icon='◈',layout='wide',initial_sidebar_state='collapsed')
    st.markdown('''<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root{--ink:#203b38;--muted:#72817c;--paper:#f4f3ee;--line:#e0e5df;--teal:#197b73}
    .stApp{background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif}
    [data-testid="stHeader"]{background:transparent}
    .block-container{max-width:1400px;padding:2.8rem 3rem 4rem}
    h1,h2,h3{font-family:'Manrope',sans-serif!important;color:var(--ink);letter-spacing:-.045em}
    h1{font-size:2.7rem!important;font-weight:800!important}
    p{line-height:1.65} .stCaption{color:var(--muted)}
    .brand{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:0 0 22px;margin-bottom:18px;gap:20px}
    .brand-name{font:800 23px 'Manrope',sans-serif;letter-spacing:.16em}.brand-name span{color:var(--teal)}
    .brand-detail{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-top:5px}
    .author{text-align:right;font-size:12px;color:var(--muted)}
    div[data-testid="stRadio"]>div[role="radiogroup"]{gap:7px;background:#eaece5;padding:6px;border-radius:15px;display:flex;flex-wrap:wrap}
    div[data-testid="stRadio"] label{padding:9px 15px!important;border-radius:10px;margin:0!important;transition:background .18s}
    div[data-testid="stRadio"] label:has(input:checked){background:#fff;box-shadow:0 2px 5px #203b380c}
    div[data-testid="stRadio"] label p{font-size:13px;font-weight:600}
    .eyebrow{font-size:11px;color:var(--teal);letter-spacing:.18em;text-transform:uppercase;font-weight:700;margin-top:30px}
    .subtitle{max-width:780px;color:var(--muted);font-size:15px;margin-bottom:22px}
    div[data-testid="stMetric"]{padding:20px 24px;background:#fff;border:1px solid var(--line);border-radius:17px;min-height:120px}
    div[data-testid="stMetricValue"]{font-family:'Manrope',sans-serif;color:var(--ink);font-size:29px}
    div[data-testid="stMetricLabel"]{color:var(--muted)}
    div[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:20px;padding:8px;overflow:hidden}
    div[data-testid="stExpander"]{background:#fff;border:1px solid var(--line);border-radius:14px}
    div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}
    .stButton button,.stDownloadButton button{border-radius:10px;border:1px solid #cfd9d1;font-weight:600;min-height:42px}
    .stButton button[kind="primary"]{background:var(--teal);border-color:var(--teal);color:white}
    .stButton button:hover,.stDownloadButton button:hover{border-color:var(--teal);color:var(--teal)}
    .stButton button[kind="primary"]:hover{color:white;background:#125d57}
    div[data-testid="stAlert"]{border-radius:12px;background:#eaf1eb}
    .feature{background:#fff;border:1px solid var(--line);border-radius:18px;padding:25px;min-height:220px;margin:12px 0}
    .feature .index{font:500 12px 'Manrope';color:#aa8b60;letter-spacing:.12em}
    .feature h3{font-size:22px!important;margin:12px 0 8px}.feature p{color:var(--muted);font-size:14px;margin:0}
    .note{border-left:2px solid #b79358;padding:12px 20px;color:var(--muted);font-size:13px;margin-top:20px}
    @media(max-width:800px){.block-container{padding:2rem 1rem}.brand{align-items:flex-start}.author{max-width:140px}h1{font-size:2rem!important}div[data-testid="stRadio"] label{padding:6px!important}}
    @media(prefers-reduced-motion:reduce){*{transition:none!important}}
    </style>''',unsafe_allow_html=True)
    st.markdown(f'''<div class="brand"><div><div class="brand-name"><span>◈</span> ATLAS</div><div class="brand-detail">Laboratoire des courbes de taux</div></div><div class="author">{html.escape(author)}<br>ENSA Agadir · Finance & ingénierie décisionnelle</div></div>''',unsafe_allow_html=True)

def page_heading(i):
    titles=['Comprendre la courbe. Explorer les modèles.','Du prix au taux zéro-coupon.','Dessiner une courbe continue.','Comparer deux signatures de marché.','Explorer la dynamique des taux.','Estimer à partir des observations.']
    descriptions=['Un espace de travail pour reconstruire, ajuster et simuler la structure des taux d’intérêt.','Reconstruisez les facteurs d’actualisation, maturité après maturité, puis contrôlez les prix obtenus.','Explorez les effets de l’interpolation et du lissage sur les facteurs d’actualisation.','Ajustez Nelson–Siegel et Svensson aux mêmes prix pour comparer leurs courbes et leurs erreurs.','Modifiez les hypothèses et observez leurs effets sur les trajectoires et la courbe de valorisation.','Utilisez les moments conditionnels pour estimer les paramètres historiques de Vasicek ou de CIR.']
    st.markdown(f'<div class="eyebrow">ATLAS / {i+1:02d} — '+['Vue générale','Construction','Interpolation','Modèles paramétriques','Simulation','Calibration'][i]+'</div>',unsafe_allow_html=True)
    st.title(titles[i])
    st.markdown(f'<div class="subtitle">{descriptions[i]}</div>',unsafe_allow_html=True)

def plot(xs,series,title,ylabel):
    fig=go.Figure()
    for j,(label,ys) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=xs,y=ys,name=label,mode='lines',line=dict(color=COLORS[j%len(COLORS)],width=2.8 if len(series)<4 else 1.7),hovertemplate='%{y:.4f}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title=dict(text=title,font=dict(size=17,color='#203b38')),template='plotly_white',height=440,paper_bgcolor='white',plot_bgcolor='white',font=dict(family='DM Sans, Arial',color='#72817c',size=12),margin=dict(l=28,r=25,t=65,b=55),hovermode='x unified',hoverlabel=dict(bgcolor='#203b38',font_color='white'),legend=dict(orientation='h',y=-.2,x=0),xaxis_title='Maturité / temps (années)',yaxis_title=ylabel)
    fig.update_xaxes(showgrid=False,zeroline=False,linecolor='#e6ebe5')
    fig.update_yaxes(gridcolor='#eef1ec',zeroline=False)
    st.plotly_chart(fig,width='stretch',config={'displaylogo':False,'scrollZoom':False,'toImageButtonOptions':{'format':'png','scale':2}})

def overview(options,demo,bootstrap,chart):
    d=demo();b=bootstrap(d);rates=-np.log(b)/d.maturity.to_numpy()*100
    cols=st.columns(3)
    cols[0].metric('Taux zéro-coupon · 1 an',f'{rates[0]:.2f} %')
    cols[1].metric('Taux zéro-coupon · 20 ans',f'{rates[-1]:.2f} %')
    cols[2].metric('Écart 20 ans − 1 an',f'{(rates[-1]-rates[0])*100:.0f} pb')
    st.caption('Démonstration synthétique · 20 obligations · taux continus · aucune cotation réelle')
    chart(d.maturity,{'Taux zéro-coupon':rates},'La structure des taux en un regard')
    st.markdown('### Choisissez votre point de départ')
    def go_to(page):st.session_state.navigation=page
    for col,index,title,desc,label in zip(st.columns(3),[1,3,5],['Construire','Comparer','Estimer'],['Des prix obligataires aux facteurs d’actualisation. Vérifiez chaque étape du bootstrap.','Confrontez Nelson–Siegel et Svensson sur un même jeu de données.','Retrouvez les paramètres historiques à partir d’une série de taux.'],['Ouvrir le bootstrap →','Comparer les modèles →','Lancer une estimation →']):
        with col:
            st.markdown(f'<div class="feature"><div class="index">0{index} / EXPLORER</div><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
            st.button(label,key=f'goto{index}',on_click=go_to,args=(options[index],),width='stretch')
    st.markdown('<div class="note">Les paramètres estimés sur des données historiques relèvent de P. La valorisation utilise des paramètres sous Q : l’application distingue ces deux usages.</div>',unsafe_allow_html=True)
