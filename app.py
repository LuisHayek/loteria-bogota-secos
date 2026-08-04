import streamlit as st
import pandas as pd
import numpy as np
import os, sys

# Intentar importar matplotlib, si falla usar graficos nativos de streamlit
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except:
    HAS_MPL = False

st.set_page_config(page_title="Lotería Bogotá - Secos Predictor", page_icon="🎰", layout="wide")

st.title("🎰 Lotería de Bogotá - Predictor de Premios Secos")
st.markdown("**Análisis estadístico real 21.852 secos (2014-2022). No garantiza ganar, pero juegas con datos, no a ciegas.**")

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file, sep=';', encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv(file, sep=';', encoding='latin1')
        except:
            df = pd.read_csv(file, sep=',', encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    df['FECHA_PARSED'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
    df['SORTEO_NUM'] = pd.to_numeric(df['SORTEO'], errors='coerce')
    df_secos = df[df['NOMBRE_PREMIO'].str.contains('SECO', case=False, na=False)].copy()
    df_secos = df_secos.sort_values(['FECHA_PARSED','SORTEO_NUM'])
    df_secos['NUMERO_PAD'] = df_secos['NUMERO'].astype(int).apply(lambda x: f"{x:04d}")
    df_secos['D1'] = df_secos['NUMERO_PAD'].str[0].astype(int)
    df_secos['D2'] = df_secos['NUMERO_PAD'].str[1].astype(int)
    df_secos['D3'] = df_secos['NUMERO_PAD'].str[2].astype(int)
    df_secos['D4'] = df_secos['NUMERO_PAD'].str[3].astype(int)
    df_secos['SUMA'] = df_secos['D1']+df_secos['D2']+df_secos['D3']+df_secos['D4']
    return df, df_secos

# Buscar archivo por defecto en varias rutas (para que funcione en cloud)
posibles_rutas = ["data/loteriabogota.csv", "loteriabogota.csv", "uploads/loteriabogota.csv", "/home/user/uploads/loteriabogota.csv"]
default_path = None
for p in posibles_rutas:
    if os.path.exists(p):
        default_path = p
        break

st.sidebar.header("📁 Datos")
uploaded = st.sidebar.file_uploader("Sube tu CSV actualizado (2023-2025 opcional)", type=["csv"])

df = None
df_secos = None

if uploaded:
    df, df_secos = load_data(uploaded)
    st.sidebar.success(f"CSV subido: {len(df)} filas")
elif default_path:
    df, df_secos = load_data(default_path)
    st.sidebar.success(f"Usando: {default_path} -> {len(df_secos)} secos")
else:
    st.error("No encuentro loteriabogota.csv. Súbelo con el botón de la izquierda o asegúrate que esté en data/loteriabogota.csv en GitHub.")
    st.stop()

ultima_fecha = df_secos['FECHA_PARSED'].max()
ano_antes = ultima_fecha - pd.DateOffset(years=1)
df_ult = df_secos[df_secos['FECHA_PARSED'] >= ano_antes]

st.sidebar.header("⚙️ Configuración")
num_pred = st.sidebar.slider("¿Cuántos números generar?", 3, 30, 6)
sum_min, sum_max = st.sidebar.slider("Filtro suma (recomendado 14-23)", 0, 36, (14,23))

freq_D1 = df_secos['D1'].value_counts().sort_index()
freq_D2 = df_secos['D2'].value_counts().sort_index()
freq_D3 = df_secos['D3'].value_counts().sort_index()
freq_D4 = df_secos['D4'].value_counts().sort_index()
freq_D1_u = df_ult['D1'].value_counts().sort_index()
freq_D2_u = df_ult['D2'].value_counts().sort_index()
freq_D3_u = df_ult['D3'].value_counts().sort_index()
freq_D4_u = df_ult['D4'].value_counts().sort_index()

def prob_dict(freq):
    total=freq.sum()
    return {k:v/total for k,v in freq.items()}

pD1, pD2, pD3, pD4 = map(prob_dict, [freq_D1, freq_D2, freq_D3, freq_D4])
pD1u, pD2u, pD3u, pD4u = map(prob_dict, [freq_D1_u, freq_D2_u, freq_D3_u, freq_D4_u])
freq_num = df_secos['NUMERO_PAD'].value_counts().to_dict()
todos_numeros = set([f"{i:04d}" for i in range(10000)])
nunca = sorted(list(todos_numeros - set(freq_num.keys())))
serie_hot = df_secos['SERIE'].value_counts().head(5).index.tolist()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔥 Frecuencias", "🏆 Top", "🎯 Generador"])

with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Secos", f"{len(df_secos):,}")
    c2.metric("Sorteos", df_secos['SORTEO_NUM'].nunique())
    c3.metric("Distintos", f"{len(freq_num)} / 10000")
    c4.metric("Nunca salidos", len(nunca))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Suma de dígitos")
        if HAS_MPL:
            fig, ax = plt.subplots()
            df_secos['SUMA'].hist(bins=36, ax=ax, edgecolor='black')
            ax.set_xlabel("Suma"); ax.set_ylabel("Frecuencia")
            st.pyplot(fig)
        else:
            st.bar_chart(df_secos['SUMA'].value_counts().sort_index())
        st.caption(f"Media {df_secos['SUMA'].mean():.2f} | Moda {df_secos['SUMA'].mode().values[0]} | 70% entre 14-23")
    with col2:
        st.subheader("Pares vs Impares")
        pares = (df_secos['NUMERO']%2==0).sum()
        st.metric("Pares", f"{pares} ({pares/len(df_secos)*100:.1f}%)")
        st.metric("Impares", f"{len(df_secos)-pares} ({(len(df_secos)-pares)/len(df_secos)*100:.1f}%)")

with tab2:
    st.subheader("Frecuencia por posición - Histórico")
    if HAS_MPL:
        fig, axes = plt.subplots(2,2, figsize=(10,6))
        axes[0,0].bar(freq_D1.index, freq_D1.values); axes[0,0].set_title('Miles D1')
        axes[0,1].bar(freq_D2.index, freq_D2.values); axes[0,1].set_title('Centenas D2')
        axes[1,0].bar(freq_D3.index, freq_D3.values); axes[1,0].set_title('Decenas D3')
        axes[1,1].bar(freq_D4.index, freq_D4.values); axes[1,1].set_title('Unidades D4')
        st.pyplot(fig)
    else:
        c1,c2 = st.columns(2)
        c1.bar_chart(freq_D1)
        c2.bar_chart(freq_D2)
    st.dataframe(pd.DataFrame({"D1":freq_D1,"D2":freq_D2,"D3":freq_D3,"D4":freq_D4}))

with tab3:
    st.subheader("Top 20 más salidos")
    top20 = pd.Series(freq_num).sort_values(ascending=False).head(20)
    st.bar_chart(top20)
    st.dataframe(top20.reset_index().rename(columns={'index':'NUMERO','NUMERO_PAD':'VECES'}))
    st.subheader("Series calientes")
    st.dataframe(df_secos['SERIE'].value_counts().head(10))

with tab4:
    st.header("Generador Inteligente")
    st.markdown("Score = 0.6*Global + 0.4*Último año + bonus suma 17-20")

    @st.cache_data
    def generar_scores(_pD1,_pD2,_pD3,_pD4,_pD1u,_pD2u,_pD3u,_pD4u):
        lista=[]
        for i in range(10000):
            s=f"{i:04d}"
            d1,d2,d3,d4=map(int, list(s))
            sg = _pD1.get(d1,0)+_pD2.get(d2,0)+_pD3.get(d3,0)+_pD4.get(d4,0)
            sr = _pD1u.get(d1,0)+_pD2u.get(d2,0)+_pD3u.get(d3,0)+_pD4u.get(d4,0)
            score = 0.6*sg + 0.4*sr
            suma=d1+d2+d3+d4
            if 17<=suma<=20:
                score*=1.15
            elif 14<=suma<=23:
                score*=1.05
            else:
                score*=0.85
            lista.append((s,score,suma))
        return sorted(lista, key=lambda x: x[1], reverse=True)

    scores = generar_scores(pD1,pD2,pD3,pD4,pD1u,pD2u,pD3u,pD4u)
    filtrados=[]
    for s,sc,sm in scores:
        if not (sum_min <= sm <= sum_max):
            continue
        filtrados.append((s,sc,sm,freq_num.get(s,0)))

    final=[]
    for s,sc,sm,v in filtrados[:num_pred*2]:
        if len(final)>=num_pred:
            break
        final.append({"NUMERO":s,"SCORE":round(sc,5),"SUMA":sm,"VECES":v,"SERIE":serie_hot[len(final)%len(serie_hot)] if serie_hot else 0})

    df_final = pd.DataFrame(final[:num_pred])
    st.dataframe(df_final, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    for i, col in enumerate([c1,c2,c3]):
        if i < len(df_final):
            row=df_final.iloc[i]
            with col:
                st.markdown(f"<div style='border:2px solid gold;border-radius:15px;padding:15px;background:#1a237e;color:white;text-align:center'><h2>{row['NUMERO']}</h2><p>SERIE {row['SERIE']}<br>SUMA {row['SUMA']} | Veces {row['VECES']}</p></div>", unsafe_allow_html=True)

    csv = df_final.to_csv(index=False, sep=';').encode('utf-8')
    st.download_button("📥 Descargar CSV", csv, "predicciones.csv", "text/csv")
