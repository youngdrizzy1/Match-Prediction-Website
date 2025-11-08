import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Premier League Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    model_gf = joblib.load('model_gf_simple.pkl')
    model_ga = joblib.load('model_ga_simple.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    team_stats = joblib.load('team_stats.pkl')
    return model_gf, model_ga, feature_cols, team_stats

try:
    model_gf, model_ga, feature_cols, team_stats = load_models()
    teams_list = sorted(team_stats.keys())
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Title
st.title("⚽ Premier League Match Predictor")
st.markdown("### Predict match outcomes using Machine Learning")
st.divider()

# Team selection
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    home_team = st.selectbox("🏠 Home Team", teams_list, index=0)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)

with col3:
    away_team = st.selectbox("✈️ Away Team", teams_list, index=1)

# Predict button
if st.button("🔮 Predict Match Result", type="primary", use_container_width=True):
    
    if home_team == away_team:
        st.error("⚠️ Please select two different teams!")
    else:
        # Create feature vector for prediction
        def create_prediction_features(home_team, away_team, team_stats, feature_cols):
            """Create feature vector using team statistics"""
            
            # Get stats for both teams
            home_stats = team_stats.get(home_team, {})
            away_stats = team_stats.get(away_team, {})
            
            # Initialize feature dictionary
            features = {}
            
            # Basic match info
            features['is_home'] = 1  # Home team always 1
            features['round_num'] = 20  # Mid-season default
            features['days_since'] = 7  # Default 1 week rest
            features['season_progress'] = 20/38  # Mid-season
            
            # Default current match stats (will be ignored by model anyway)
            features['Poss'] = 50
            features['Sh'] = 12
            features['SoT'] = 4
            features['Dist'] = 17
            features['FK'] = 2
            features['PK'] = 0
            features['PKatt'] = 0
            features['Attendance'] = 50000
            
            # Home team's rolling stats
            features['GF_roll5'] = home_stats.get('GF_roll5', 1.5)
            features['GA_roll5'] = home_stats.get('GA_roll5', 1.0)
            features['xG_roll5'] = home_stats.get('xG_roll5', 1.4)
            features['xGA_roll5'] = home_stats.get('xGA_roll5', 1.0)
            features['Poss_roll5'] = home_stats.get('Poss_roll5', 50)
            features['Sh_roll5'] = home_stats.get('Sh_roll5', 12)
            features['SoT_roll5'] = home_stats.get('SoT_roll5', 4)
            features['win_rate_5'] = home_stats.get('win_rate_5', 0.4)
            features['GD_roll5'] = home_stats.get('GD_roll5', 0.5)
            features['xG_diff_roll5'] = home_stats.get('xG_diff_roll5', 0.4)
            features['shot_accuracy'] = home_stats.get('shot_accuracy', 0.33)
            
            # Away team's rolling stats (as opponent)
            features['opp_GF_roll5'] = away_stats.get('GF_roll5', 1.5)
            features['opp_GA_roll5'] = away_stats.get('GA_roll5', 1.0)
            features['opp_xG_roll5'] = away_stats.get('xG_roll5', 1.4)
            features['opp_xGA_roll5'] = away_stats.get('xGA_roll5', 1.0)
            features['opp_Poss_roll5'] = away_stats.get('Poss_roll5', 50)
            features['opp_Sh_roll5'] = away_stats.get('Sh_roll5', 12)
            features['opp_SoT_roll5'] = away_stats.get('SoT_roll5', 4)
            features['opp_win_rate_5'] = away_stats.get('win_rate_5', 0.4)
            features['opp_GD_roll5'] = away_stats.get('GD_roll5', 0.5)
            features['opp_xG_diff_roll5'] = away_stats.get('xG_diff_roll5', 0.4)
            features['opp_shot_accuracy'] = away_stats.get('shot_accuracy', 0.33)
            
            # Form difference
            features['form_diff'] = features['win_rate_5'] - features['opp_win_rate_5']
            
            # Head-to-head (default neutral)
            features['h2h_wins'] = 1.0
            features['h2h_goals_for'] = 1.5
            features['h2h_goals_against'] = 1.5
            
            # Streak features (default neutral)
            features['streak'] = 0.0
            features['home_form'] = features['win_rate_5']
            features['away_form'] = features['opp_win_rate_5']
            
            # Create DataFrame with correct column order
            feature_df = pd.DataFrame([features])
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            # Return in correct order
            return feature_df[feature_cols]
        
        # Create features
        X_pred = create_prediction_features(home_team, away_team, team_stats, feature_cols)
        
        # Make predictions
        gf_pred = model_gf.predict(X_pred)[0]
        ga_pred = model_ga.predict(X_pred)[0]
        
        # Round to whole numbers
        gf_final = int(round(gf_pred))
        ga_final = int(round(ga_pred))
        
        # Determine result with draw threshold
        goal_diff = gf_pred - ga_pred
        
        if abs(goal_diff) < 0.4:
            result = "Draw"
            result_emoji = "🤝"
            result_color = "orange"
        elif gf_final > ga_final:
            result = f"{home_team} Win"
            result_emoji = "🏆"
            result_color = "green"
        elif gf_final < ga_final:
            result = f"{away_team} Win"
            result_emoji = "🏆"
            result_color = "red"
        else:
            result = "Draw"
            result_emoji = "🤝"
            result_color = "orange"
        
        # Display results
        st.divider()
        st.markdown("## 📊 Prediction Results")
        
        # Main prediction box
        st.markdown(f"""
        <div style='background-color: {result_color}; padding: 30px; border-radius: 10px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>{result_emoji} {result}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Score prediction
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3>{home_team}</h3>
                <h1 style='color: #1f77b4; font-size: 60px; margin: 10px 0;'>{gf_final}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h1 style='text-align: center; margin-top: 50px;'>-</h1>", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3>{away_team}</h3>
                <h1 style='color: #ff7f0e; font-size: 60px; margin: 10px 0;'>{ga_final}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Team stats comparison
        st.markdown("### 📈 Team Form Comparison")
        
        col1, col2 = st.columns(2)
        
        home_stats = team_stats[home_team]
        away_stats = team_stats[away_team]
        
        with col1:
            st.markdown(f"**{home_team} (Recent Form)**")
            st.metric("Goals Scored (avg)", f"{home_stats['GF_roll5']:.2f}")
            st.metric("Goals Conceded (avg)", f"{home_stats['GA_roll5']:.2f}")
            st.metric("Win Rate", f"{home_stats['win_rate_5']*100:.1f}%")
            st.metric("Goal Difference", f"{home_stats['GD_roll5']:.2f}")
        
        with col2:
            st.markdown(f"**{away_team} (Recent Form)**")
            st.metric("Goals Scored (avg)", f"{away_stats['GF_roll5']:.2f}")
            st.metric("Goals Conceded (avg)", f"{away_stats['GA_roll5']:.2f}")
            st.metric("Win Rate", f"{away_stats['win_rate_5']*100:.1f}%")
            st.metric("Goal Difference", f"{away_stats['GD_roll5']:.2f}")
        
        # Model confidence
        st.divider()
        st.markdown("### 🎯 Prediction Confidence")
        
        confidence = 100 - (abs(goal_diff) * 20)  # Simple confidence metric
        confidence = max(50, min(95, confidence))  # Cap between 50-95%
        
        st.progress(confidence / 100)
        st.markdown(f"**Confidence Level:** {confidence:.1f}%")
        
        st.info("""
        💡 **Note:** Predictions are based on recent team performance (last 5 matches). 
        Actual results may vary due to injuries, tactics, and other factors not captured by the model.
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>⚽ Built with Random Forest ML Model | Trained on Premier League Data (2017-2025)</p>
</div>
""", unsafe_allow_html=True)