import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px


st.markdown("""
<style>

/* SFONDO PAGINA (opzionale ma utile) */
.stApp {
    background-color: #1E1E1E;
    color: #ffffff;
}

/* TITOLI */
h1, h2, h3 {
    text-align: center;
}

/* BOTTONI (se li usi) */
.stButton>button {
    background-color: #00cc88;
    color: black;
    border-radius: 8px;
}

/* METRICHE */
[data-testid="stMetricValue"] {
}

/* DATAFRAME header */
thead tr th {
}

/* DIVIDER */
hr {
    opacity: 0.6;
}

/* CONTAINER (bordo vero visibile) */
div[data-testid="stContainer"] {
    border: 1px solid rgba(0, 204, 136, 0.2);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}
            
/* SLIDER - TRACK (parte colorata) */
.stSlider > div > div > div > div {
    background: #9370D8;
}


</style>
""", unsafe_allow_html=True)

######################################## TITOLO

st.write("")
st.write("")

st.set_page_config(
    page_title="Forecasting Bologna - Air Quality",
    layout="wide"
)

st.write("")
st.write("")

st.markdown("""
<div style="
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 5px;
    letter-spacing: 1px;
">
    Junior Data Science Academy 2026 • CRIF • GiGroup
</div>
""", unsafe_allow_html=True)

st.write("")

st.markdown("""
<div style="
    text-align: center;
    font-size: 14px;
    color: gray;
    margin-bottom: 5px;
    letter-spacing: 1px;
">
    Un progetto sviluppato da Giorgia Martinelli - Francesco Pascullo - Susanna Casarini
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='color: #22c55e;'>Forecasting Bologna - Air Quality</h1>",
    unsafe_allow_html=True
)

st.write("")

######################################## PROGETTO


with st.container():

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.markdown(
            """
            <p style='font-size:18px; text-align: justify;'>
            Il progetto si concentra sull'analisi del PM10 a Bologna, con particolare attenzione agli episodi in cui viene superata la soglia di 50 µg/m³ nel biennio 2024–2025. I dati provengono da tre centraline di rilevamento collocate in punti strategici della città.
            </p>
            """,
            unsafe_allow_html=True
        )

st.divider()

######################################## DATI

st.markdown(
    "<h2 style='color: #facc15;'>Dati sulla qualità dell'aria e meteo</h2>",
    unsafe_allow_html=True
)

st.write("")
st.write("")

col1, col2 = st.columns(2)

with col1:

######################################## MAPPA

    data = pd.DataFrame({
        "lat": [44.483768, 44.500289, 44.499788],
        "lon": [11.355158, 11.328987, 11.286281],
        "centralina": ["Giardini Margherita", "San Felice", "Via Chiarini"]
    })

    st.write("""Stazioni fisse di monitoraggio della qualità dell'aria a Bologna""")


    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position='[lon, lat]',
        get_color='[255, 220, 0]',
        get_radius=80,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=44.49,
        longitude=11.32,
        zoom=12
    )

    tooltip = {
        "text": "{centralina}"
    }

    deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip
        )

    st.pydeck_chart(
            deck,
            use_container_width=True,
            height=420
        )



######################################## DATASET

with col2:

    df = pd.read_parquet('df.parquet')

    # fai vedere la mappa di bologna con le centraline

    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:

            st.write("""
            Dataset sul PM10 e sul meteo per Bologna.
            Include variabili ambientali e climatiche usate
            per prevedere il superamento del PM10.
            """)

        with col2:
            st.metric("Osservazioni", df.shape[0])
            
        with col3:
            st.metric("Variabili", df.shape[1])

        st.dataframe(df.head(50), use_container_width=True, height=350)

st.write("")
st.write("")

######################################## GRAFICO PM10 NEL TEMPO

st.write("""Andamento giornaliero del PM10 a Bologna""")

x = df["giorno"]
y = df["PM10"]

sopra = y > 50

fig = go.Figure()

# punti sotto soglia
fig.add_trace(go.Scatter(
    x=x[~sopra],
    y=y[~sopra],
    mode="markers",
    name="PM10 ≤ 50",
    marker=dict(color="gray", size=6)
))

# punti sopra soglia
fig.add_trace(go.Scatter(
    x=x[sopra],
    y=y[sopra],
    mode="markers",
    name="PM10 > 50",
    marker=dict(color="red", size=7)
))

# linea soglia
fig.add_hline(y=50, line_dash="dash", line_color="gray")

# layout scuro + compatto
fig.update_layout(
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    height=450,
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)


######################################## VARIABILI

st.write("")
st.write("")

import plotly.express as px

df_var = df.copy()

list_var = [
    'dew_point_2m_max',
    'precipitation_sum',
    'cloud_cover_max',
    'wind_speed_10m_min',
    'wind_gusts_10m_max',
    'pressure_msl_max',
    'soil_moisture_28_to_100cm_mean',
    'temperature_2m_max',
    'shortwave_radiation_sum'
]

st.write("""Relazione tra le variabili e la target PM10""")

col1, col2 = st.columns([1, 2])

with col1:

    var = st.radio(
        "Seleziona variabile",
        list_var,
        label_visibility="collapsed"
    )

with col2:
    fig = px.violin(
        df_var,
        x="target_pm10",
        y=var,
        color="target_pm10",
        color_discrete_map={0: "#f2f2f2", 1: "#ff2d2d"},
        points=False
    )

    fig.update_traces(
        spanmode="soft"
    )

    fig.update_layout(
        height=320,  # leggermente più compatto
        margin=dict(l=5, r=5, t=25, b=5),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

####################################### MODELLO

st.write("")
st.write("")

with st.container():
    st.markdown(
        "<h2 style='color: #00CED1;'>Modello Predittivo</h2>",
        unsafe_allow_html=True)

st.write("")
st.write("")


col1, col2 = st.columns([1,4])

with col2:

    with st.container():

        _, center, _ = st.columns([1, 3, 1])

        with center: 
            st.markdown(
                """
                <h4 style='color: #00CED1;'>XGBoost</h4>

                <div style='font-size:18px; line-height:1.6; text-align: justify';>
                <p>
                Abbiamo utilizzato un modello XGBoost, una tecnica di machine learning basata su gradient boosting che combina alberi decisionali in sequenza,
                correggendo progressivamente gli errori. È efficace per dati tabellari come quelli meteorologici e ambientali grazie alla capacità di modellare relazioni non lineari.
                </p>

                <h4 style='color: #00CED1;'>Recall come metrica principale</h4>

                <p>
                Il modello è stato ottimizzato per massimizzare la recall, così da individuare il più possibile i superamenti del PM10.
                In questo contesto è preferibile ridurre i falsi negativi, anche a costo di qualche falso positivo, dato il carattere preventivo delle decisioni.
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )

with col1:

    tab_modello = pd.read_csv("modello.csv")

    st.markdown("Misure di performance del modello")

    cols_to_show = ["Recall", "Precision", "F1-score", "MCC", "Accuracy"]

    df = tab_modello[cols_to_show].T
    df.index.name = "Metriche"

    st.dataframe(
        df,
        use_container_width=True
    )

st.divider()

################################ FEATURE SHAP
df = pd.read_parquet('df.parquet')

model = joblib.load("xgb_model.pkl")

list_var = [
    'precipitation_sum',
    'cloud_cover_max',
    'wind_speed_10m_min',
    'wind_gusts_10m_max',
    'pressure_msl_max',
    'soil_moisture_28_to_100cm_mean',
    'dew_point_2m_max',
    'temperature_2m_max',
    'shortwave_radiation_sum'
]


st.set_page_config(layout="wide")



X = df[list_var]
y = df["target_pm10"]

background = X.sample(100, random_state=42)

explainer = shap.Explainer(model, background)
shap_values = explainer(X)



mean_shap = shap_values.abs.mean(0).values
max_shap = shap_values.abs.max(0).values



col1, col2 = st.columns(2)


with col1:
    shap_mean_importance = pd.DataFrame({
        "feature": X.columns,
        "mean_shap": mean_shap
    })

    # ordina per importanza crescente
    shap_mean_importance = shap_mean_importance.sort_values("mean_shap", ascending=True)

    st.subheader("Mean SHAP Importance")

    fig1 = px.bar(
        shap_mean_importance,
        x="mean_shap",
        y="feature",
        orientation="h",
        template="plotly_dark",
        color="mean_shap",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig1, use_container_width=True)



with col2:
    shap_importance = pd.DataFrame({
        "feature": X.columns,
        "max_shap": max_shap
    })

    # ordina per importanza
    shap_importance = shap_importance.sort_values("max_shap", ascending=True)

    fig2 = px.bar(
        shap_importance,
        x="max_shap",
        y="feature",
        orientation="h",
        template="plotly_dark",
        color="max_shap",
        color_continuous_scale="Blues"
    )

    st.subheader("Max Absolute SHAP")
    st.plotly_chart(fig2, use_container_width=True)



st.subheader("Feature Impact Distribution")

shap_df = pd.DataFrame(shap_values.values, columns=X.columns)

shap_long = shap_df.melt(var_name="feature", value_name="shap_value")

fig3 = px.strip(
    shap_long,
    x="shap_value",
    y="feature",
    orientation="h",
    color="feature",
    template="plotly_dark"
)

st.plotly_chart(fig3, use_container_width=True)

################################ APP


var_info = {
    'precipitation_sum': {"name": "🌧️ Precipitazioni", "unit": "mm", "effect": "⬇️ PM10 diminuisce", "desc": "La pioggia ingloba le particelle di PM10 che cade al suolo"},
    'cloud_cover_max': {"name": "☁️ Copertura nuvolosa", "unit": "%", "effect": "⬆️ PM10 aumenta", "desc": "Non c'è un ricircolo d'aria e stagna"},
    'wind_speed_10m_min': {"name": "💨 Velocità vento", "unit": "km/h", "effect": "⬇️ PM10 diminuisce", "desc": "Il vento disperde gli inquinanti"},
    'wind_gusts_10m_max': {"name": "🌪️ Raffiche vento", "unit": "km/h", "effect": "⬇️ PM10 diminuisce", "desc": "Maggiore mescolamento dell’aria, ma da considerare l'interazione con l'umidità del suolo"},
    'pressure_msl_max': {"name": "📈 Pressione", "unit": "hPa", "effect": "⬆️ PM10 aumenta", "desc": "Alta pressione promuove il ristagno"},
    'soil_moisture_28_to_100cm_mean': {"name": "🌱 Umidità suolo", "unit": "%", "effect": "⬇️ Leggera diminuzione", "desc": "Meno polveri sollevate"},
    'dew_point_2m_max': {"name": "💧 Punto di rugiada", "unit": "°C", "effect": "⬆️ Leggero aumento nella fascia tra i 5 e i 10 gradi", "desc": "Aria umida e stagnante"},
    'temperature_2m_max': {"name": "🌡️ Temperatura", "unit": "°C", "effect": "⬇️ PM10 diminuisce", "desc": "La diminuzione della temperatura favorisce accumulo inquinanti"},
    'shortwave_radiation_sum': {"name": "☀️ Radiazione solare", "unit": "MJ/m²", "effect": "⬆️ Leggero aumento", "desc": "Condizioni stabili e soleggiate"}
}



st.markdown("""
<style>

/* SFONDO */
.stApp {
    background-color: #1E1E1E;
    color: white;
}

/* CONTAINER PICCOLI */
div[data-testid="stContainer"] {
    border: 1px solid rgba(0, 204, 136, 0.25);
    border-radius: 10px;
    padding: 6px !important;
    margin-bottom: 6px;
            
}

/* RIDUZIONE SPAZI */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* SLIDER COMPATTI */
.stSlider {
    margin-top: -8px;
    margin-bottom: -8px;
}

/* TRACK SLIDER */
.stSlider > div > div > div > div {
    background: #9370D8;
}

/* PALLINO */
.stSlider [role="slider"] {
    background: #9370D8;
    border: 1px solid white;
    transform: scale(0.8);
}

/* BOTTONI GRANDI */
.stButton>button {
    background-color: #00cc88;
    color: black;
    border-radius: 12px;
    height: 60px;
    font-size: 18px;
    font-weight: 600;
}

/* RISULTATO PIÙ VISIBILE */
.stAlert {
    font-size: 18px;
    padding: 16px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


with st.container(border=True):

    st.markdown(
        "<h2 style='text-align:center; color:#9370D8;'>PM10 Prediction Tool</h2>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    user_input = {}

    cols = st.columns([1, 1, 1, 1], gap="small")

    for i, var in enumerate(list_var):

        min_val = float(min(df[var]))
        max_val = float(max(df[var]))
        mean_val = float(df[var].mean())

        info = var_info[var]

        with cols[i % 4]:

            with st.container(border=True):

                st.markdown(
                    f"""
                    <div style='font-size:13px; font-weight:700; margin-bottom:2px;'>
                        {info["name"]}
                    </div>

                    <div style='font-size:11px; color:gray; margin-bottom:4px;'>
                        {info["unit"]} • {info["effect"]}
                    </div>

                    <div style='font-size:10px; color:gray; margin-bottom:6px;'>
                        {info["desc"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                user_input[var] = st.slider(
                    label="",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=var
                )

    st.divider()


    if st.button("Previsione supermento soglia PM10 > 50 μg/m3", use_container_width=True, type="primary"):

        input_df = pd.DataFrame([user_input])
        input_df = input_df.reindex(columns=list_var)

#       proba = model.predict_proba(input_df[list_var].values)[:, 1][0]
     
        proba = model.predict_proba(input_df)[0, 1]

        risk = proba * 100

        st.metric(
            label="PM10 Risk Score",
            value=f"{risk:.1f} / 100"
        )

        st.progress(int(min(risk, 100)))

        if risk < 20:
            st.success(f"🟢 Low risk ({risk:.1f}/100)")
        elif risk < 50:
            st.warning(f"🟡 Medium risk ({risk:.1f}/100)")
        else:
            st.error(f"🔴 High risk ({risk:.1f}/100)")


st.divider()

st.markdown("### References")

st.write("")
st.write("")

st.markdown("""
<div style='text-align:center;'>
📁 Dataset e codice: <a href='https://github.com/SusannaCasarini/Forecasting-Bologna-Air-Quality/tree/main'>GitHub repository</a>
</div>
""", unsafe_allow_html=True)


st.write("")
st.write("")
