### --- Packages
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import pickle
import shap
from lime import lime_tabular
import plotly.graph_objects as go

import sklearn
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance

import json
import requests

### --- DASHBOARD 
def main():

    # Predictions
    df = load_pred()
    df = df.drop(columns=['index'])
    df = df.reset_index(drop=True)

    # Features
    features_sample = load_feat()

  # Tables
    y = features_sample[["TARGET"]]
    X = features_sample.drop(columns=["TARGET"])
    X.index = range(X.shape[0])
    
    X_id=X[["SK_ID_CURR"]]

    X = X.drop(columns=["SK_ID_CURR"])

   # model
    model = load_model()

    importance = load_globale_importance()
    
    shap_values = load_locale_importance()
    
    # Probability threshold
    proba_threshold = 0.8
    
     
    page = st.sidebar.selectbox("Choisir une page", ["Page d'accueil",
                                                     "Exploration",
                                                     "Prédiction - numéro du client",
                                                     "Prédiction - caractéristiques du client"])

    if page == "Page d'accueil":
        st.title("Tableau de bord : Accord de crédits")
        st.markdown("### Le tableau de board permet d'informer sur la solvabilité bancaire d'un client selon ses caractéristiques financières. En effet, au regard d'un seuil de probabilité, un algorithme prédictif considère un client comme étant solvable ou non.")
#    -------------------------
    elif page == "Exploration":
        st.title("Prédiction des accords de crédits des clients")
        if st.checkbox('Quelques lignes de données !'):
            tab = df.rename(columns={ 'SK_ID_CURR': 'Numéro du client',
                                       'Prediction': 'Prédiction',
                                       'Probability': 'Solvabilité',})
            st.dataframe(tab)
        
        st.markdown("### Analyse des prédictions")
        st.text("Répartition des prédictions d'accord de crédits")
        x = df.groupby(["Prediction"])["Prediction"].count()
        fig = plt.figure(figsize = (8, 8))
        plt.pie(x,
#                 labels = ["Refusé","Accepté"],
           colors = ['red', 'blue'],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.7,
            labeldistance = 1.4,
           shadow = True)
        plt.legend(["Refusé","Accepté"])
        st.pyplot(fig)
        
        st.markdown("### Interprétabilité globale du modèle")
        st.text("Distribution des caractéristiques selon leur pouvoir prédictif de solvabilité")
        ### --- Features global importance
        features_importance_names = list(X.columns)
        features_names = pd.DataFrame(features_importance_names, columns =["Caractéristiques"])
        importance_scores = pd.DataFrame(importance, columns =["Scores"])
        features_names.reset_index(drop=True, inplace=True)
        importance_scores.reset_index(drop=True, inplace=True)
        importance_table = pd.concat([features_names,importance_scores], axis=1)                # summarize feature importance
#         for i,v in enumerate(importance): 
#             print('Feature: %0d, Score: %.5f' % (i,v))
#         print(importance_table.head())
                 # Plot feature importance
        st.bar_chart(data=importance_table,
                     x="Caractéristiques",
                     y="Scores",
                     width=300,
                     height=300,
                     use_container_width=True)
        
 #    -------------------------       
    elif page == "Prédiction - numéro du client":
        st.title('Prédiction des clients existants')
        ### --- SELECTION OF CUSTOMER
        customer_number = st.selectbox("Choisir le numéro du client : ", list(X_id['SK_ID_CURR'].sort_values()))
        # En-tête
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        # URL
        url = "https://myappwithgithub.herokuapp.com/predict/" + str(customer_number)
        response = requests.get(url, headers=headers)

        try:
             proba = response.json()['Probability']
             classe = response.json()['Classe']
        except json.JSONDecodeError as identifier:
             print("Error occur", identifier.msg)

        ### --- FILTER DATAFRAME BASED ON SELECTION
        mask_customer = (df['SK_ID_CURR']==customer_number)
        st.markdown(f'*Probabilité de solvabilité: {proba:.2f}*')
        st.markdown(f'*Crédit accepté: {classe}*')
        
        ### --- PROBABILITY GAUGE
        plot_bgcolor = "#def"
        quadrant_colors = [plot_bgcolor, "#2bad4e", "#f25829"]
        quadrant_text = ["", "<b>Accepté</b>", "<b>Refusé</b>"]
        n_quadrants = len(quadrant_colors) - 1

        #current_value = float(df[mask_customer]['Probability'])
        current_value = proba
        min_value = 0
        max_value = 1
        hand_length = np.sqrt(2) / 4
        hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

        fig = go.Figure(
            data=[
                go.Pie(
                    values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                    rotation=90,
                    hole=0.5,
                    marker_colors=quadrant_colors,
                    text=quadrant_text,
                    textinfo="text",
                    hoverinfo="skip",
                ),
            ],
            layout=go.Layout(
                showlegend=False,
                margin=dict(b=0,t=10,l=10,r=10),
                width=450,
                height=450,
                paper_bgcolor=plot_bgcolor,
                annotations=[
                    go.layout.Annotation(
                        text=f"<b>Solvabilité:</b><br>{current_value}",
                        x=0.5, xanchor="center", xref="paper",
                        y=0.25, yanchor="bottom", yref="paper",
                        showarrow=False,
                    )
                ],
                shapes=[
                    go.layout.Shape(
                        type="circle",
                        x0=0.48, x1=0.52,
                        y0=0.48, y1=0.52,
                        fillcolor="#333",
                        line_color="#333",
                    ),
                    go.layout.Shape(
                        type="line",
                        x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                        y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                        line=dict(color="#333", width=4)
                    )
                ]
            )
        )
        st.plotly_chart(fig)
 
        st.markdown("### Interprétabilité local du modèle") 
        ### --- FEATURES LOCALE IMPORTANCE
        st.text("Importance des caractéristiques du client")
        # Plot summary_plot
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_id = X_id[X_id["SK_ID_CURR"] == customer_number].index[0]

        st.pyplot(shap.plots.bar(shap_values[num_id]))
        
        st.text("Distribution de l'importance d'une caractéristique selon la classe")
        # Multi_select
        features_names = X.columns.values[1:159]
        features_names_tab = pd.DataFrame(features_names,columns=["features_names"])
        caracteristic_selection = st.selectbox('Caractéristiques:',
                                             features_names)
        # Plot summary_plot
        num_f= features_names_tab[features_names_tab["features_names"]==caracteristic_selection].index[0]
        tab = pd.DataFrame(shap_values[:,num_f:num_f+1].values, columns=["Score"])
        y = y.reset_index(drop=True)
        tab1 = pd.concat([tab,y],axis=1)
        tab2 = tab1[tab1["TARGET"]==0]
        tab1 = tab1[tab1["TARGET"]==1]
        tab1 = tab1.drop(columns=["TARGET"])
        tab1 = tab1.rename(columns={"Score":"Accepté"})
        tab2 = tab2.drop(columns=["TARGET"])
        tab2 = tab2.rename(columns={"Score":"Refusé"})

        score_customer = shap_values[num_id,num_f:num_f+1].values
        
        fig = plt.figure()
        plt.hist(tab1["Accepté"], alpha=0.3, histtype='stepfilled',color='green', label='Accepté')
        plt.hist(tab2["Refusé"], alpha=0.3, histtype='stepfilled', color='red', label='Refusé')      
        plt.axvline(x=score_customer, color='blue', linestyle='--', label='Customer')
        plt.xlabel("Scores", fontsize=16)
        plt.legend()
        st.pyplot(fig)

        ### --- Map
        st.text("Représentation graphique de l'importance des caractéristiques")
        # table
        tab_select1 = pd.DataFrame(shap_values[:, 0:2].values, columns=["feat1", "feat2"])
        tab_select2 = pd.DataFrame(shap_values[num_id:num_id + 1, 0:2].values, columns=["feat1", "feat2"])
        # Plot
        fig = plt.figure()
        plt.scatter(x=tab_select1[["feat1"]],
                    y=tab_select1[["feat2"]],
                    color='blue',
                    label='Global')
        plt.scatter(x=tab_select2[["feat1"]],
                    y=tab_select2[["feat2"]],
                    color='green',
                    label='Local')
        plt.xlabel(features_names[1], fontsize=16)
        plt.ylabel(features_names[2], fontsize=16)
        plt.legend()
        st.pyplot(fig)

    #    -------------------------
    else:
        st.title("Prédiction de nouveau client")
        #uploaded_file = st.file_uploader("Importer les caractéristiques du client (fichier CSV).")
        input_vector = st.text_input("Entrer les caractéristiques du client.",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder)
        ### --- SELECTION OF VECTOR
        if input_vector is not None:
            headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
            feature = json.dumps(input_vector)

            url = "https://myappwithgithub.herokuapp.com/predict_model"
            response = requests.post(url, data=feature, headers=headers)
            try:
                proba = response.json()['Probability']
                classe = response.json()['Classe']
            except json.JSONDecodeError as identifier:
                print("Error occur", identifier.msg)

            # --- MODEL APPLICATION
            st.markdown(f'*Probabilité de solvabilité: {proba}*')
            st.markdown(f'*Crédit accepté: {classe}*')

#             feature = pd.read_csv(uploaded_file, delimiter=";")
#             # --- MODEL APPLICATION
#             probability = model.predict_proba(feature)
#             # I dont'understand
#             probability = str(probability)
#             probability = probability.split()[0]
#             probability = probability.replace("[[", "")
#             probability = float(probability)
#             decision = "OUI" if probability > proba_threshold else "NON"
#             st.markdown(f'*Probabilité de solvabilité: {probability}*')
#             st.markdown(f'*Crédit accepté: {decision}*')

            ### --- PROBABILITY GAUGE
            plot_bgcolor = "#def"
            quadrant_colors = [plot_bgcolor, "#2bad4e", "#f25829"]
            quadrant_text = ["", "<b>Accepté</b>", "<b>Refusé</b>"]
            n_quadrants = len(quadrant_colors) - 1

            #current_value = probability
            current_value = proba
            min_value = 0
            max_value = 1
            hand_length = np.sqrt(2) / 4
            hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

            fig = go.Figure(
                data=[
                    go.Pie(
                        values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                        rotation=90,
                        hole=0.5,
                        marker_colors=quadrant_colors,
                        text=quadrant_text,
                        textinfo="text",
                        hoverinfo="skip",
                    ),
                ],
                layout=go.Layout(
                    showlegend=False,
                    margin=dict(b=0,t=10,l=10,r=10),
                    width=450,
                    height=450,
                    paper_bgcolor=plot_bgcolor,
                    annotations=[
                        go.layout.Annotation(
                            text=f"<b>Solvabilité:</b><br>{current_value}",
                            x=0.5, xanchor="center", xref="paper",
                            y=0.25, yanchor="bottom", yref="paper",
                            showarrow=False,
                        )
                    ],
                    shapes=[
                        go.layout.Shape(
                            type="circle",
                            x0=0.48, x1=0.52,
                            y0=0.48, y1=0.52,
                            fillcolor="#333",
                            line_color="#333",
                        ),
                        go.layout.Shape(
                            type="line",
                            x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                            y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                            line=dict(color="#333", width=4)
                        )
                    ]
                )
            )
            st.plotly_chart(fig)

# @st.cache(allow_output_mutation=True)
@st.cache
# Load data into the dataframe.
def load_feat():
    return pickle.load(open('df_features_sample.pkl','rb'))
def load_pred():
    return pickle.load(open('df_modelisation.pkl','rb'))
def load_globale_importance():
    return pickle.load(open('globale_importance.pkl','rb'))
def load_locale_importance():
    return pickle.load(open('locale_importance.pkl','rb'))
def load_model():
    return pickle.load(open('model.pkl','rb'))

if __name__ == '__main__': 
    main()
