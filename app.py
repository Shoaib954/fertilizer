import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="AI Farm Advisor", layout="wide", page_icon="🌱")


def get_base_models():
    return {
        'LogisticRegression':         LogisticRegression(max_iter=1000, random_state=42),
        'GaussianNB':                 GaussianNB(),
        'SVC':                        SVC(probability=True, random_state=42),
        'KNeighborsClassifier':       KNeighborsClassifier(n_neighbors=5),
        'DecisionTreeClassifier':     DecisionTreeClassifier(random_state=42),
        'ExtraTreeClassifier':        ExtraTreeClassifier(random_state=42),
        'RandomForestClassifier':     RandomForestClassifier(n_estimators=100, random_state=42),
        'BaggingClassifier':          BaggingClassifier(random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'AdaBoostClassifier':         AdaBoostClassifier(random_state=42),
        'CatBoostClassifier':         CatBoostClassifier(verbose=0, random_state=42),
        'LGBMClassifier':             LGBMClassifier(verbose=-1, random_state=42),
    }


@st.cache_resource
def train_models():
    # ── Irrigation ──────────────────────────────────────────────────────────
    df_irr = pd.read_csv('Datasets/Irrigation_recommendation.csv')
    le_irr = LabelEncoder().fit(df_irr['irrigation_schedule'])
    X_irr = df_irr.drop('irrigation_schedule', axis=1)
    y_irr = le_irr.transform(df_irr['irrigation_schedule'])
    X_tr, X_te, y_tr, y_te = train_test_split(X_irr, y_irr, test_size=0.3, random_state=42, stratify=y_irr)

    irr_base = get_base_models()
    for m in irr_base.values():
        m.fit(X_tr, y_tr)
    irr_stack = StackingClassifier(
        estimators=list(irr_base.items()),
        final_estimator=LogisticRegression(max_iter=1000, random_state=42), cv=5
    )
    irr_stack.fit(X_tr, y_tr)

    # ── Fertilizer (new 10000-row dataset) ──────────────────────────────────
    df_fert = pd.read_csv('Datasets/fertilizer_recommendation.csv')

    cat_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
                'Irrigation_Type', 'Previous_Crop', 'Region']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder().fit(df_fert[col])
        encoders[col] = le
        df_fert[col] = le.transform(df_fert[col])

    le_fert = LabelEncoder().fit(df_fert['Recommended_Fertilizer'])
    X_fert = df_fert.drop('Recommended_Fertilizer', axis=1)
    y_fert = le_fert.transform(df_fert['Recommended_Fertilizer'])
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_fert, y_fert, test_size=0.3, random_state=42, stratify=y_fert)

    fert_base = get_base_models()
    for m in fert_base.values():
        m.fit(X_tr2, y_tr2)
    fert_stack = StackingClassifier(
        estimators=list(fert_base.items()),
        final_estimator=LogisticRegression(max_iter=1000, random_state=42), cv=5
    )
    fert_stack.fit(X_tr2, y_tr2)

    return irr_base, irr_stack, le_irr, fert_base, fert_stack, le_fert, encoders


def predict(stack_model, le_target, input_df):
    idx = stack_model.predict(input_df)[0]
    proba = stack_model.predict_proba(input_df)[0]
    label = le_target.inverse_transform([int(idx)])[0]
    return label, proba, le_target.classes_


def individual_preds(base_models, le_target, input_df):
    rows = []
    for name, m in base_models.items():
        raw = m.predict(input_df)[0]
        label = le_target.inverse_transform([int(raw)])[0]
        rows.append({'Model': name, 'Prediction': label})
    return pd.DataFrame(rows)


def get_ai_response(messages, context, api_key):
    client = Groq(api_key=api_key)
    system_prompt = f"""You are an expert agricultural AI assistant specializing in irrigation scheduling and fertilizer recommendations.
{f'Current prediction context: {context}' if context else ''}
Keep answers concise, practical and farmer-friendly. Use bullet points where helpful."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        max_tokens=512, temperature=0.7
    )
    return response.choices[0].message.content


# ── Load models ──────────────────────────────────────────────────────────────
with st.spinner("🌱 Training AI models... please wait"):
    irr_base, irr_stack, le_irr, fert_base, fert_stack, le_fert, encoders = train_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("Images/shell.webp", width=280)
page = st.sidebar.radio("Navigate", [
    "💧 Irrigation Recommendation",
    "🌿 Fertilizer Recommendation",
    "🤖 AI Farm Assistant"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Stacking Ensemble** — 12 ML models")
st.sidebar.markdown(f"Irrigation classes: {', '.join(le_irr.classes_)}")
st.sidebar.markdown(f"Fertilizer classes: {len(le_fert.classes_)} types")
st.sidebar.markdown("---")
groq_api_key = st.sidebar.text_input("🔑 Groq API Key", type="password",
                                      placeholder="gsk_...", help="Free API key from console.groq.com")


# ════════════════════════════════════════════════════════════════════════════
# IRRIGATION PAGE
# ════════════════════════════════════════════════════════════════════════════
if page == "💧 Irrigation Recommendation":
    st.title("💧 Irrigation Recommendation")
    st.markdown("Enter your soil and climate parameters to get the best irrigation schedule.")

    with st.form("irr_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            N           = st.slider('Nitrogen (N)', 0, 140, 50)
            P           = st.slider('Phosphorus (P)', 5, 145, 50)
            K           = st.slider('Potassium (K)', 5, 205, 50)
        with c2:
            temperature = st.slider('Temperature (°C)', 8.0, 44.0, 25.0, 0.1)
            humidity    = st.slider('Humidity (%)', 14.0, 100.0, 65.0, 0.1)
        with c3:
            ph          = st.slider('pH Level', 3.5, 10.0, 6.5, 0.1)
            rainfall    = st.slider('Rainfall (mm)', 20.0, 300.0, 100.0, 0.1)
        submitted = st.form_submit_button("💧 Get Irrigation Recommendation", type="primary")

    if submitted:
        irr_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        prediction, proba, classes = predict(irr_stack, le_irr, irr_input)
        confidence = proba.max() * 100

        st.success(f"💧 **{prediction}** irrigation schedule recommended with **{confidence:.1f}%** confidence")

        col_a, col_b = st.columns(2)
        with col_a:
            prob_df = pd.DataFrame({'Schedule': [str(c) for c in classes], 'Probability': proba}).sort_values('Probability')
            fig = px.bar(prob_df, x='Probability', y='Schedule', orientation='h',
                         title='Prediction Probabilities', color='Probability', color_continuous_scale='blues')
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fi = irr_base['RandomForestClassifier'].feature_importances_
            labels = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            fi_df = pd.DataFrame({'Feature': labels, 'Importance': fi}).sort_values('Importance', ascending=False)
            fig2 = px.bar(fi_df, x='Feature', y='Importance', title='Feature Importance (RF)',
                          color='Importance', color_continuous_scale='greens')
            st.plotly_chart(fig2, use_container_width=True)

        raw_vals = [float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall)]
        fig3 = go.Figure([go.Bar(x=labels, y=raw_vals, marker_color='steelblue')])
        fig3.update_layout(title='Your Input Values', xaxis_title='Feature', yaxis_title='Value')
        st.plotly_chart(fig3, use_container_width=True)

        schedule_info = {
            'Daily':         ('🔴', 'High urgency',  'Water every day. Very low rainfall and high temperature detected.'),
            'Weekly':        ('🟠', 'Moderate-high', 'Water once a week. Monitor soil conditions closely.'),
            'Bi-weekly':     ('🟡', 'Moderate',      'Water every two weeks. Conditions are manageable.'),
            'Monthly':       ('🟢', 'Low',           'Water once a month. Good rainfall and humidity present.'),
            'No Irrigation': ('🔵', 'Minimal',       'No irrigation needed. Rainfall is sufficient.'),
        }
        icon, urgency, advice = schedule_info.get(prediction, ('⚪', 'Unknown', ''))

        st.markdown("---")
        st.markdown("## 📋 Final Recommendation Report")
        col1, col2, col3 = st.columns(3)
        col1.metric("Irrigation Schedule", f"{icon} {prediction}")
        col2.metric("Urgency Level", urgency)
        col3.metric("AI Confidence", f"{confidence:.1f}%")
        st.info(f"**What to do:** {advice}")

        tips = []
        if rainfall < 60:    tips.append("⚠️ Very low rainfall — ensure irrigation system is active.")
        if temperature > 35: tips.append("☀️ High temperature — increase irrigation frequency.")
        if humidity < 40:    tips.append("🌵 Low humidity — consider drip irrigation.")
        if rainfall > 220:   tips.append("🌧️ Heavy rainfall — skip irrigation to avoid waterlogging.")
        if ph < 5.5:         tips.append("🧪 Acidic soil — consider liming before irrigation.")
        if ph > 7.5:         tips.append("🧪 Alkaline soil — monitor nutrient availability.")
        if N < 20:           tips.append("🌱 Low Nitrogen — apply N-rich fertilizer alongside irrigation.")
        if tips:
            st.markdown("**💡 Additional Insights:**")
            for tip in tips:
                st.markdown(f"- {tip}")

        with st.expander("📊 Individual Model Predictions", expanded=False):
            st.dataframe(individual_preds(irr_base, le_irr, irr_input), use_container_width=True, hide_index=True)

        csv = pd.DataFrame([{
            'N': N, 'P': P, 'K': K, 'Temperature (°C)': temperature,
            'Humidity (%)': humidity, 'pH': ph, 'Rainfall (mm)': rainfall,
            'Recommended Schedule': prediction, 'Confidence (%)': round(confidence, 2)
        }]).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report as CSV", data=csv,
                           file_name="irrigation_result.csv", mime="text/csv")

        st.session_state['irr_context'] = (
            f"Irrigation: {prediction} ({confidence:.1f}% confidence). "
            f"N:{N}, P:{P}, K:{K}, Temp:{temperature}°C, Humidity:{humidity}%, pH:{ph}, Rainfall:{rainfall}mm."
        )


# ════════════════════════════════════════════════════════════════════════════
# FERTILIZER PAGE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🌿 Fertilizer Recommendation":
    st.title("🌿 Fertilizer Recommendation")
    st.markdown("Set your soil and crop parameters to get an AI-based fertilizer suggestion.")

    with st.form("fert_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            soil_ph      = st.slider('Soil pH', 4.0, 9.0, 6.5, 0.1)
            soil_moisture= st.slider('Soil Moisture (%)', 10.0, 60.0, 30.0, 0.1)
            organic_c    = st.slider('Organic Carbon (%)', 0.1, 5.0, 1.5, 0.1)
            elec_cond    = st.slider('Electrical Conductivity', 0.1, 4.0, 1.0, 0.1)
        with c2:
            nitrogen     = st.slider('Nitrogen Level', 0, 100, 40)
            phosphorus   = st.slider('Phosphorus Level', 0, 100, 40)
            potassium    = st.slider('Potassium Level', 0, 100, 40)
            temperature  = st.slider('Temperature (°C)', 10, 45, 28)
            humidity     = st.slider('Humidity (%)', 20, 100, 55)
            rainfall     = st.slider('Rainfall (mm)', 0.0, 300.0, 100.0, 0.1)
        with c3:
            soil_type    = st.selectbox('Soil Type', list(encoders['Soil_Type'].classes_))
            crop_type    = st.selectbox('Crop Type', list(encoders['Crop_Type'].classes_))
            growth_stage = st.selectbox('Crop Growth Stage', list(encoders['Crop_Growth_Stage'].classes_))
            season       = st.selectbox('Season', list(encoders['Season'].classes_))
            irr_type     = st.selectbox('Irrigation Type', list(encoders['Irrigation_Type'].classes_))
            prev_crop    = st.selectbox('Previous Crop', list(encoders['Previous_Crop'].classes_))
            region       = st.selectbox('Region', list(encoders['Region'].classes_))
            fert_last    = st.slider('Fertilizer Used Last Season (kg)', 0.0, 200.0, 50.0, 0.5)
            yield_last   = st.slider('Yield Last Season (tons)', 0.0, 10.0, 3.0, 0.1)
        submitted_f = st.form_submit_button("🌿 Get Fertilizer Recommendation", type="primary")

    if submitted_f:
        fert_input = pd.DataFrame([[
            encoders['Soil_Type'].transform([soil_type])[0],
            soil_ph, soil_moisture, organic_c, elec_cond,
            nitrogen, phosphorus, potassium,
            temperature, humidity, rainfall,
            encoders['Crop_Type'].transform([crop_type])[0],
            encoders['Crop_Growth_Stage'].transform([growth_stage])[0],
            encoders['Season'].transform([season])[0],
            encoders['Irrigation_Type'].transform([irr_type])[0],
            encoders['Previous_Crop'].transform([prev_crop])[0],
            encoders['Region'].transform([region])[0],
            fert_last, yield_last
        ]], columns=[
            'Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
            'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level',
            'Temperature', 'Humidity', 'Rainfall',
            'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type',
            'Previous_Crop', 'Region', 'Fertilizer_Used_Last_Season', 'Yield_Last_Season'
        ])

        fert_pred, fert_proba, fert_classes = predict(fert_stack, le_fert, fert_input)
        fert_conf = fert_proba.max() * 100

        st.success(f"🌿 **{fert_pred}** recommended with **{fert_conf:.1f}%** confidence")

        col_a, col_b = st.columns(2)
        with col_a:
            prob_df = pd.DataFrame({'Fertilizer': [str(c) for c in fert_classes],
                                    'Probability': fert_proba}).sort_values('Probability')
            fig = px.bar(prob_df, x='Probability', y='Fertilizer', orientation='h',
                         title='Prediction Probabilities', color='Probability', color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fi = fert_base['RandomForestClassifier'].feature_importances_
            f_labels = ['Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
                        'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level',
                        'Temperature', 'Humidity', 'Rainfall',
                        'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type',
                        'Previous_Crop', 'Region', 'Fertilizer_Used_Last_Season', 'Yield_Last_Season']
            fi_df = pd.DataFrame({'Feature': f_labels, 'Importance': fi}).sort_values('Importance', ascending=False).head(10)
            fig2 = px.bar(fi_df, x='Feature', y='Importance', title='Top 10 Feature Importance (RF)',
                          color='Importance', color_continuous_scale='oranges')
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

        fert_info = {
            'Urea':          ('🟡', 'High Nitrogen',   'Best for leafy growth. Apply during vegetative stage.'),
            'DAP':           ('🟠', 'High N+P',        'Ideal for root development and early crop growth.'),
            'MOP':           ('🔵', 'High Potassium',  'Improves fruit quality and disease resistance.'),
            'SSP':           ('🟣', 'High Phosphorus', 'Promotes strong root system and flowering.'),
            'NPK':           ('🟢', 'Balanced N-P-K',  'All-purpose fertilizer for general crop nutrition.'),
            'Compost':       ('🟤', 'Organic',         'Improves soil structure and long-term fertility.'),
            'Zinc Sulphate': ('⚪', 'Micronutrient',   'Corrects zinc deficiency. Apply before sowing.'),
        }
        icon, type_label, advice = fert_info.get(fert_pred, ('⚪', 'General', 'Follow standard application guidelines.'))

        st.markdown("---")
        st.markdown("## 📋 Final Recommendation Report")
        col1, col2, col3 = st.columns(3)
        col1.metric("Recommended Fertilizer", f"{icon} {fert_pred}")
        col2.metric("Type", type_label)
        col3.metric("AI Confidence", f"{fert_conf:.1f}%")
        st.info(f"**Application advice:** {advice}")

        tips = []
        if nitrogen < 20:    tips.append("🌱 Low Nitrogen — crop may show yellowing leaves.")
        if phosphorus < 20:  tips.append("🌿 Low Phosphorus — poor root development expected.")
        if potassium < 20:   tips.append("🍂 Low Potassium — crop susceptible to disease.")
        if soil_moisture < 20: tips.append("💧 Low soil moisture — irrigate before fertilizer application.")
        if soil_type == 'Sandy': tips.append("🏜️ Sandy soil — nutrients leach quickly. Use split applications.")
        if growth_stage == 'Flowering': tips.append("🌸 Flowering stage — avoid high nitrogen, focus on P and K.")
        if tips:
            st.markdown("**💡 Additional Insights:**")
            for tip in tips:
                st.markdown(f"- {tip}")

        with st.expander("📊 Individual Model Predictions", expanded=False):
            st.dataframe(individual_preds(fert_base, le_fert, fert_input), use_container_width=True, hide_index=True)

        csv = pd.DataFrame([{
            'Soil_Type': soil_type, 'Soil_pH': soil_ph, 'Soil_Moisture': soil_moisture,
            'Organic_Carbon': organic_c, 'Electrical_Conductivity': elec_cond,
            'Nitrogen': nitrogen, 'Phosphorus': phosphorus, 'Potassium': potassium,
            'Temperature': temperature, 'Humidity': humidity, 'Rainfall': rainfall,
            'Crop_Type': crop_type, 'Growth_Stage': growth_stage, 'Season': season,
            'Irrigation_Type': irr_type, 'Previous_Crop': prev_crop, 'Region': region,
            'Recommended Fertilizer': fert_pred, 'Confidence (%)': round(fert_conf, 2)
        }]).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report as CSV", data=csv,
                           file_name="fertilizer_result.csv", mime="text/csv")

        st.session_state['fert_context'] = (
            f"Fertilizer: {fert_pred} ({fert_conf:.1f}% confidence) for {crop_type} on {soil_type} soil. "
            f"N:{nitrogen}, P:{phosphorus}, K:{potassium}, Season:{season}."
        )


# ════════════════════════════════════════════════════════════════════════════
# AI CHATBOT PAGE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Farm Assistant":
    st.title("🤖 AI Farm Assistant")
    st.markdown("Ask me anything about irrigation, fertilizers, soil health, or crop management.")

    if not groq_api_key:
        st.warning("⚠️ Please enter your **Groq API Key** in the sidebar. Get a free key at [console.groq.com](https://console.groq.com)")
        st.stop()

    context = " | ".join([st.session_state[k] for k in ['irr_context', 'fert_context'] if k in st.session_state])
    if context:
        st.info(f"💡 Context from your last predictions: {context}")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    user_input = st.chat_input("Ask about irrigation, fertilizers, soil, crops...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_response(st.session_state.chat_history, context, groq_api_key)
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    if not st.session_state.chat_history:
        st.markdown("**💬 Try asking:**")
        suggestions = [
            "What irrigation schedule suits low rainfall areas?",
            "When should I use Urea vs DAP fertilizer?",
            "How does sandy soil affect irrigation frequency?",
            "What is the ideal pH for wheat cultivation?",
            "How do I know if my crop needs more potassium?",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            if col.button(suggestion, use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": suggestion})
                response = get_ai_response(st.session_state.chat_history, context, groq_api_key)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
