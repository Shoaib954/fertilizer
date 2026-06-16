import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Farm Advisor", layout="wide", page_icon="🌱")

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    irr  = joblib.load("Models/Irrigation_Recommendation_model.pkl")
    fert = joblib.load("Models/Fertilizer_Recommendation_model.pkl")
    return irr, fert

irr_bundle, fert_bundle = load_models()

irr_base     = irr_bundle['base_models']
irr_meta     = irr_bundle['meta_model']
le_soil_irr  = irr_bundle['le_soil']
le_crop_irr  = irr_bundle['le_crop']
le_sched     = irr_bundle['le_target']

fert_base    = fert_bundle['base_models']
fert_meta    = fert_bundle['meta_model']
le_soil_fert = fert_bundle['le_soil']
le_crop_fert = fert_bundle['le_crop']
le_fert      = fert_bundle['le_target']

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict(base_models, meta_model, le_target, input_df):
    base_preds = np.hstack([m.predict_proba(input_df) for m in base_models.values()])
    idx        = meta_model.predict(base_preds)[0]
    proba      = meta_model.predict_proba(base_preds)[0]
    label      = le_target.inverse_transform([int(idx)])[0]
    return label, proba, le_target.classes_

def individual_preds(base_models, le_target, input_df):
    rows = []
    for name, m in base_models.items():
        raw = m.predict(input_df)[0]
        rows.append({'Model': str(name), 'Prediction': str(le_target.inverse_transform([int(raw)])[0])})
    return pd.DataFrame(rows)

# ── Groq AI chatbot ───────────────────────────────────────────────────────────
def get_ai_response(messages, context="", api_key=""):
    client = Groq(api_key=api_key)
    system_prompt = f"""You are an expert agricultural AI assistant specializing in irrigation scheduling and fertilizer recommendations. 
You help farmers make data-driven decisions about crop management.
You have deep knowledge of:
- Soil types (Sandy, Loamy, Clayey, Black, Red) and their water retention
- Irrigation schedules (Daily, Weekly, Bi-weekly, Monthly, As-needed)
- Fertilizers (Urea, DAP, MOP, TSP, NPK blends) and their use cases
- Crops: Rice, Wheat, Maize, Cotton, Sugarcane, Paddy, Barley, etc.
- Evapotranspiration, soil moisture, pH, rainfall effects on irrigation
- Nitrogen, Phosphorous, Potassium roles in plant growth

{f'Current prediction context: {context}' if context else ''}

Keep answers concise, practical and farmer-friendly. Use bullet points where helpful."""

    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=full_messages,
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].message.content

# ── Final recommendation card ─────────────────────────────────────────────────
def show_irrigation_recommendation(prediction, confidence, soil_moisture, evapotranspiration,
                                    temperature, humidity, rainfall, ph, soil_type, crop_type):
    schedule_info = {
        'Daily':     ('🔴', 'High urgency', 'Water every day. Soil is critically dry or evapotranspiration is very high.'),
        'Weekly':    ('🟠', 'Moderate-high', 'Water once a week. Monitor soil moisture closely.'),
        'Bi-weekly': ('🟡', 'Moderate', 'Water every two weeks. Conditions are manageable.'),
        'Monthly':   ('🟢', 'Low', 'Water once a month. Good rainfall or high soil moisture present.'),
        'As-needed': ('🔵', 'Minimal', 'Irrigate only when soil moisture drops. Conditions are favorable.'),
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
    if soil_moisture < 30:
        tips.append("⚠️ Soil moisture is critically low — irrigate immediately regardless of schedule.")
    if evapotranspiration > 8:
        tips.append("☀️ High evapotranspiration detected — increase water frequency.")
    if rainfall > 200:
        tips.append("🌧️ Heavy rainfall recorded — reduce or skip irrigation this cycle.")
    if ph < 6.0:
        tips.append("🧪 Acidic soil (pH < 6) — consider liming before irrigation.")
    if ph > 7.5:
        tips.append("🧪 Alkaline soil (pH > 7.5) — monitor nutrient availability.")
    if soil_type == 'Sandy':
        tips.append("🏜️ Sandy soil drains fast — consider more frequent, smaller irrigation doses.")
    if soil_type == 'Clayey':
        tips.append("🏔️ Clayey soil retains water — avoid over-irrigation and waterlogging.")
    if crop_type == 'Rice':
        tips.append("🌾 Rice requires standing water — ensure paddy fields are adequately flooded.")
    if crop_type == 'Cotton':
        tips.append("🌿 Cotton is drought-tolerant — avoid excess moisture to prevent root rot.")

    if tips:
        st.markdown("**💡 Additional Insights:**")
        for tip in tips:
            st.markdown(f"- {tip}")

    return f"Irrigation: {prediction} ({confidence:.1f}% confidence) for {crop_type} on {soil_type} soil. " \
           f"Soil moisture: {soil_moisture}%, Rainfall: {rainfall}mm, Temp: {temperature}°C."


def show_fertilizer_recommendation(prediction, confidence, temp_f, humidity_f, moisture_f,
                                    nitrogen, potassium, phosphorous, soil_type_f, crop_type_f):
    fert_info = {
        'Urea':     ('🟡', 'High Nitrogen', 'Best for leafy growth. Apply during vegetative stage.'),
        'DAP':      ('🟠', 'High N+P', 'Ideal for root development and early crop growth.'),
        'MOP':      ('🔵', 'High Potassium', 'Improves fruit quality, disease resistance and water uptake.'),
        'TSP':      ('🟣', 'High Phosphorus', 'Promotes strong root system and flowering.'),
        '14-35-14': ('🟢', 'Balanced N-P-K', 'Good for flowering and fruiting stages.'),
        '28-28':    ('🔴', 'High N+P', 'Suitable for high-demand crops at early growth.'),
        '17-17-17': ('⚪', 'Equal N-P-K', 'All-purpose fertilizer for general crop nutrition.'),
        '20-20':    ('🟤', 'Balanced', 'Good for soil with moderate nutrient levels.'),
        '10-26-26': ('🌸', 'High P+K', 'Boosts root strength and grain/fruit filling.'),
    }
    icon, type_label, advice = fert_info.get(prediction, ('⚪', 'General', 'Follow standard application guidelines.'))

    st.markdown("---")
    st.markdown("## 📋 Final Recommendation Report")
    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended Fertilizer", f"{icon} {prediction}")
    col2.metric("Type", type_label)
    col3.metric("AI Confidence", f"{confidence:.1f}%")

    st.info(f"**Application advice:** {advice}")

    tips = []
    if nitrogen > 150:
        tips.append("⚠️ Very high Nitrogen — risk of nutrient burn. Consider split application.")
    if nitrogen < 20:
        tips.append("🌱 Low Nitrogen — crop may show yellowing (chlorosis). Apply N-rich fertilizer.")
    if phosphorous < 20:
        tips.append("🌿 Low Phosphorus — poor root development expected. Apply P-rich fertilizer.")
    if potassium < 20:
        tips.append("🍂 Low Potassium — crop susceptible to disease. Apply K-rich fertilizer.")
    if moisture_f < 30:
        tips.append("💧 Low soil moisture — irrigate before fertilizer application for better absorption.")
    if soil_type_f == 'Sandy':
        tips.append("🏜️ Sandy soil — nutrients leach quickly. Use slow-release or split applications.")
    if crop_type_f in ['Paddy', 'Rice']:
        tips.append("🌾 Paddy/Rice — apply fertilizer in standing water for uniform distribution.")

    if tips:
        st.markdown("**💡 Additional Insights:**")
        for tip in tips:
            st.markdown(f"- {tip}")

    return f"Fertilizer: {prediction} ({confidence:.1f}% confidence) for {crop_type_f} on {soil_type_f} soil. " \
           f"N:{nitrogen}, P:{phosphorous}, K:{potassium}."


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("Images/shell.webp", use_container_width=True)
page = st.sidebar.radio("Navigate", [
    "💧 Irrigation Recommendation",
    "🌿 Fertilizer Recommendation",
    "🤖 AI Farm Assistant"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Stacking Ensemble** — 12 ML models")
st.sidebar.markdown(f"Irrigation classes: {', '.join(le_sched.classes_)}")
st.sidebar.markdown(f"Fertilizer classes: {', '.join(le_fert.classes_)}")
st.sidebar.markdown("---")
groq_api_key = st.sidebar.text_input(
    "🔑 Groq API Key",
    type="password",
    placeholder="gsk_...",
    help="Free API key from console.groq.com → API Keys"
)


# ════════════════════════════════════════════════════════════════════════════
# IRRIGATION PAGE
# ════════════════════════════════════════════════════════════════════════════
if page == "💧 Irrigation Recommendation":
    st.title("💧 Irrigation Recommendation")
    st.markdown("Set your field parameters and click **Get Recommendation** for an AI-based irrigation schedule.")

    with st.form("irr_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            soil_moisture      = st.slider('Soil Moisture (%)', 10, 90, 50)
            evapotranspiration = st.slider('Evapotranspiration (mm)', 1.0, 12.0, 5.0, 0.1)
            temperature        = st.slider('Temperature (°C)', 15, 45, 28)
        with c2:
            humidity  = st.slider('Humidity (%)', 30, 95, 55)
            rainfall  = st.slider('Rainfall (mm)', 0, 350, 100)
            ph        = st.slider('pH Level', 5.5, 8.5, 6.8, 0.1)
        with c3:
            soil_type = st.selectbox('Soil Type', list(le_soil_irr.classes_))
            crop_type = st.selectbox('Crop Type', list(le_crop_irr.classes_))
        submitted = st.form_submit_button("💧 Get Irrigation Recommendation", type="primary", use_container_width=True)

    if submitted:
        irr_input = pd.DataFrame([[
            soil_moisture, evapotranspiration, temperature, humidity, rainfall,
            ph, le_soil_irr.transform([soil_type])[0], le_crop_irr.transform([crop_type])[0]
        ]], columns=['soil_moisture','evapotranspiration','temperature','humidity','rainfall','ph','soil_type','crop_type'])

        with st.spinner("Running AI prediction..."):
            prediction, proba, classes = predict(irr_base, irr_meta, le_sched, irr_input)
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
            labels = ['Soil Moisture','Evapotranspiration','Temperature','Humidity','Rainfall','pH','Soil Type','Crop Type']
            fi_df = pd.DataFrame({'Feature': labels, 'Importance': fi}).sort_values('Importance', ascending=False)
            fig2 = px.bar(fi_df, x='Feature', y='Importance', title='Feature Importance (RF)',
                          color='Importance', color_continuous_scale='greens')
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

        raw_vals = [float(soil_moisture), float(evapotranspiration), float(temperature),
                    float(humidity), float(rainfall), float(ph)]
        fig3 = go.Figure([go.Bar(x=labels[:6], y=raw_vals, marker_color='steelblue')])
        fig3.update_layout(title='Your Input Values', xaxis_title='Feature', yaxis_title='Value')
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📊 Individual Model Predictions", expanded=False):
            st.dataframe(individual_preds(irr_base, le_sched, irr_input), use_container_width=True)

        context = show_irrigation_recommendation(
            prediction, confidence, soil_moisture, evapotranspiration,
            temperature, humidity, rainfall, ph, soil_type, crop_type
        )

        csv = pd.DataFrame([{
            'Soil Moisture (%)': soil_moisture, 'Evapotranspiration (mm)': evapotranspiration,
            'Temperature (°C)': temperature, 'Humidity (%)': humidity,
            'Rainfall (mm)': rainfall, 'pH': ph,
            'Soil Type': soil_type, 'Crop Type': crop_type,
            'Recommended Schedule': prediction, 'Confidence (%)': round(confidence, 2)
        }]).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report as CSV", data=csv,
                           file_name="irrigation_result.csv", mime="text/csv")

        st.session_state['irr_context'] = context


# ════════════════════════════════════════════════════════════════════════════
# FERTILIZER PAGE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🌿 Fertilizer Recommendation":
    st.title("🌿 Fertilizer Recommendation")
    st.markdown("Set your soil and crop parameters and click **Get Recommendation** for an AI-based fertilizer suggestion.")

    with st.form("fert_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            temp_f     = st.slider('Temperature (°C)', 15, 45, 28)
            humidity_f = st.slider('Humidity (%)', 30, 90, 55)
            moisture_f = st.slider('Moisture (%)', 20, 80, 50)
        with c2:
            nitrogen    = st.slider('Nitrogen (N)', 0, 300, 80)
            potassium   = st.slider('Potassium (K)', 0, 300, 40)
            phosphorous = st.slider('Phosphorous (P)', 0, 300, 40)
        with c3:
            soil_type_f = st.selectbox('Soil Type', list(le_soil_fert.classes_))
            crop_type_f = st.selectbox('Crop Type', list(le_crop_fert.classes_))
        submitted_f = st.form_submit_button("🌿 Get Fertilizer Recommendation", type="primary", use_container_width=True)

    if submitted_f:
        fert_input = pd.DataFrame([[
            temp_f, humidity_f, moisture_f,
            le_soil_fert.transform([soil_type_f])[0],
            le_crop_fert.transform([crop_type_f])[0],
            nitrogen, potassium, phosphorous
        ]], columns=['Temperature','Humidity','Moisture','Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous'])

        with st.spinner("Running AI prediction..."):
            fert_pred, fert_proba, fert_classes = predict(fert_base, fert_meta, le_fert, fert_input)
        fert_conf = fert_proba.max() * 100

        st.success(f"🌿 **{fert_pred}** recommended with **{fert_conf:.1f}%** confidence")

        col_a, col_b = st.columns(2)
        with col_a:
            prob_df = pd.DataFrame({'Fertilizer': [str(c) for c in fert_classes], 'Probability': fert_proba}).sort_values('Probability')
            fig = px.bar(prob_df, x='Probability', y='Fertilizer', orientation='h',
                         title='Prediction Probabilities', color='Probability', color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fi = fert_base['RandomForestClassifier'].feature_importances_
            f_labels = ['Temperature','Humidity','Moisture','Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous']
            fi_df = pd.DataFrame({'Feature': f_labels, 'Importance': fi}).sort_values('Importance', ascending=False)
            fig2 = px.bar(fi_df, x='Feature', y='Importance', title='Feature Importance (RF)',
                          color='Importance', color_continuous_scale='oranges')
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

        raw_vals = [float(temp_f), float(humidity_f), float(moisture_f),
                    float(nitrogen), float(potassium), float(phosphorous)]
        fig3 = go.Figure([go.Bar(x=['Temperature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous'],
                                  y=raw_vals, marker_color='darkorange')])
        fig3.update_layout(title='Your Input Values', xaxis_title='Feature', yaxis_title='Value')
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📊 Individual Model Predictions", expanded=False):
            st.dataframe(individual_preds(fert_base, le_fert, fert_input), use_container_width=True)

        context = show_fertilizer_recommendation(
            fert_pred, fert_conf, temp_f, humidity_f, moisture_f,
            nitrogen, potassium, phosphorous, soil_type_f, crop_type_f
        )

        csv = pd.DataFrame([{
            'Temperature (°C)': temp_f, 'Humidity (%)': humidity_f, 'Moisture (%)': moisture_f,
            'Soil Type': soil_type_f, 'Crop Type': crop_type_f,
            'Nitrogen (N)': nitrogen, 'Potassium (K)': potassium, 'Phosphorous (P)': phosphorous,
            'Recommended Fertilizer': fert_pred, 'Confidence (%)': round(fert_conf, 2)
        }]).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report as CSV", data=csv,
                           file_name="fertilizer_result.csv", mime="text/csv")

        st.session_state['fert_context'] = context


# ════════════════════════════════════════════════════════════════════════════
# AI CHATBOT PAGE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Farm Assistant":
    st.title("🤖 AI Farm Assistant")
    st.markdown("Ask me anything about irrigation, fertilizers, soil health, or crop management.")

    if not groq_api_key:
        st.warning("⚠️ Please enter your **Groq API Key** in the sidebar to use the chatbot. Get a free key at [console.groq.com](https://console.groq.com)")
        st.stop()

    # Build context from last predictions
    context_parts = []
    if 'irr_context' in st.session_state:
        context_parts.append(st.session_state['irr_context'])
    if 'fert_context' in st.session_state:
        context_parts.append(st.session_state['fert_context'])
    context = " | ".join(context_parts) if context_parts else ""

    if context:
        st.info(f"💡 I have context from your last predictions: {context}")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Chat input
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

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("**💬 Try asking:**")
        suggestions = [
            "What does soil moisture below 30% mean for my crops?",
            "When should I use Urea vs DAP fertilizer?",
            "How does sandy soil affect irrigation frequency?",
            "What is evapotranspiration and why does it matter?",
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
