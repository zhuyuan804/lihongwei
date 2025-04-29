import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shap
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Methanol Selectivity Prediction System",
    page_icon="🧪",
    layout="wide"
)

# Feature list with ranges
FEATURES = {
    'Specific surface area(m2/g)': {'min': 50, 'max': 300, 'default': 150},
    'Cu(wt%)': {'min': 30, 'max': 80, 'default': 60},
    'Pressure (Mpa)': {'min': 3, 'max': 8, 'default': 5},
    'GHSV (h-1)': {'min': 500, 'max': 5000, 'default': 2000},
    'Temp(℃)': {'min': 180, 'max': 300, 'default': 250},
    'Reduction temp(℃)': {'min': 200, 'max': 400, 'default': 300},
    'Calcination temp(℃)': {'min': 300, 'max': 600, 'default': 400},
    'Types of Cu salts': {'options': ['Cu(NO3)2', 'CuCl2', 'Cu(CH3COO)2'], 'default': 'Cu(NO3)2'},
    'Types of Zn salts': {'options': ['Zn(NO3)2', 'ZnCl2', 'Zn(CH3COO)2'], 'default': 'Zn(NO3)2'},
    'Zn(wt%)': {'min': 20, 'max': 50, 'default': 30},
    'Others-In(wt%)': {'min': 0, 'max': 20, 'default': 5},
    'Others-Zr(wt%)': {'min': 0, 'max': 20, 'default': 5},
    'Others-Al(wt%)': {'min': 0, 'max': 20, 'default': 5},
    'Others-Si(wt%)': {'min': 0, 'max': 20, 'default': 5},
    'Dry temp(℃)': {'min': 80, 'max': 150, 'default': 100},
    'Aging temp(℃)': {'min': 80, 'max': 150, 'default': 100},
    'H2 content of redcution gas(%)': {'min': 2, 'max': 20, 'default': 5},
    'Average pore size(nm)': {'min': 5, 'max': 30, 'default': 15}
}

def preprocess_input_data(input_df, feature_order=None):
    """Preprocess input data by encoding categorical variables and ensuring feature order"""
    # Create a copy of the input data
    processed_df = input_df.copy()
    
    # Convert categorical columns to category dtype
    categorical_features = ['Types of Cu salts', 'Types of Zn salts']
    for col in categorical_features:
        processed_df[col] = processed_df[col].astype('category')
    
    # Reorder features if order is provided
    if feature_order is not None:
        processed_df = processed_df[feature_order]
    
    return processed_df

def encode_categorical_features(df):
    """Encode categorical features using label encoding"""
    # Create a copy of the dataframe
    df_encoded = df.copy()
    
    # Encoding mappings
    encodings = {
        'Types of Cu salts': {
            'Cu(NO3)2': 0,
            'CuCl2': 1,
            'Cu(CH3COO)2': 2
        },
        'Types of Zn salts': {
            'Zn(NO3)2': 0,
            'ZnCl2': 1,
            'Zn(CH3COO)2': 2
        }
    }
    
    # Apply encodings
    for col, mapping in encodings.items():
        df_encoded[col] = df_encoded[col].map(mapping)
    
    return df_encoded

def load_model_and_data():
    """Load model, scaler and training data"""
    try:
        # Load model and scaler from the specified path
        model_path = 'best_model.joblib'
        scaler_path = 'scaler.joblib'
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None, None, None, None, None
            
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found at: {scaler_path}")
            return None, None, None, None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load training data
        data = pd.read_excel('data/data.xlsx')
        # Create a copy of the features to avoid SettingWithCopyWarning
        X = data[list(FEATURES.keys())].copy()
        
        # Encode categorical features
        X = encode_categorical_features(X)
        y = data['CH3OH selectivity(C%)'].copy()
        
        st.success("Model, scaler and data loaded successfully!")
        return model, scaler, X, y, list(X.columns)
    except Exception as e:
        st.error(f"Error loading model and data: {str(e)}")
        return None, None, None, None, None

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot using plotly"""
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=importance['Importance'],
        y=importance['Feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=600
    )
    
    return fig

def create_shap_analysis(model, input_data, feature_names):
    """Create SHAP analysis for the input sample"""
    try:
        # Encode categorical features for SHAP analysis
        input_data_encoded = encode_categorical_features(input_data)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_data_encoded)
        
        # Get the SHAP values array
        if isinstance(shap_values, list):
            shap_values_array = shap_values[0]
        else:
            shap_values_array = shap_values
            
        # Ensure shap_values_array is 1D
        if shap_values_array.ndim > 1:
            shap_values_array = shap_values_array.flatten()
            
        # Ensure we have the correct number of features
        if len(shap_values_array) != len(feature_names):
            st.error(f"Mismatch between SHAP values ({len(shap_values_array)}) and features ({len(feature_names)})")
            return None, None, None
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'value': input_data.iloc[0].values,
            'shap_value': shap_values_array,
            'abs_shap_value': np.abs(shap_values_array)
        })
        
        # Sort by absolute SHAP value for better visualization
        feature_importance = feature_importance.sort_values('abs_shap_value', ascending=False)
        
        # Create waterfall plot
        waterfall_fig = go.Figure()
        
        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[0]
            
        # Calculate cumulative impact using numpy array
        shap_values_sorted = feature_importance['shap_value'].values
        cumulative_impact = np.cumsum(shap_values_sorted)
        final_prediction = base_value + cumulative_impact[-1]
        
        # Create bars for waterfall plot
        x_pos = list(range(len(feature_importance) + 2))  # +2 for base value and final prediction
        y_values = np.concatenate([[base_value], base_value + cumulative_impact, [final_prediction]])
        
        # Add base value bar
        waterfall_fig.add_trace(go.Bar(
            x=[0],
            y=[base_value],
            name='Base Value',
            marker_color='blue',
            width=0.6,
            text=[f'Base<br>{base_value:.2f}'],
            textposition='outside'
        ))
        
        # Add feature contribution bars
        for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
            waterfall_fig.add_trace(go.Bar(
                x=[i],
                y=[row['shap_value']],
                base=[y_values[i]],
                name=row['feature'],
                marker_color='red' if row['shap_value'] > 0 else 'blue',
                width=0.6,
                text=[f"{row['feature']}<br>{row['value']}<br>Impact: {row['shap_value']:.4f}"],
                textposition='outside'
            ))
        
        # Add final prediction bar
        waterfall_fig.add_trace(go.Bar(
            x=[len(feature_importance) + 1],
            y=[final_prediction],
            name='Final Prediction',
            marker_color='green',
            width=0.6,
            text=[f'Final<br>{final_prediction:.2f}'],
            textposition='outside'
        ))
        
        # Update waterfall plot layout
        waterfall_fig.update_layout(
            title={
                'text': 'Feature Contribution Waterfall Plot',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            xaxis_title='Features',
            yaxis_title='Prediction Value',
            height=600,
            showlegend=False,
            bargap=0,
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                ticktext=['Base'] + list(feature_importance['feature']) + ['Final'],
                tickvals=x_pos,
                tickangle=45
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black'
            )
        )
        
        # Create heatmap data
        heatmap_data = pd.DataFrame({
            'Feature': feature_importance['feature'],
            'Value': feature_importance['value'].astype(str),
            'Impact': feature_importance['shap_value']
        })
        
        # Identify categorical and numerical features
        categorical_features = ['Types of Cu salts', 'Types of Zn salts']
        numerical_features = [f for f in feature_names if f not in categorical_features]
        
        # Create normalized values for the heatmap
        z_values = np.zeros((len(heatmap_data), 2))
        
        # Process feature values (first column)
        for idx, (feature, value) in enumerate(zip(heatmap_data['Feature'], heatmap_data['Value'])):
            if feature in categorical_features:
                z_values[idx, 0] = 0.5
            else:
                try:
                    numeric_value = float(value)
                    feature_values = heatmap_data['Value'][heatmap_data['Feature'].isin(numerical_features)].astype(float)
                    feature_min = float(feature_values.min())
                    feature_max = float(feature_values.max())
                    if feature_max > feature_min:
                        z_values[idx, 0] = (numeric_value - feature_min) / (feature_max - feature_min)
                    else:
                        z_values[idx, 0] = 0.5
                except (ValueError, TypeError):
                    z_values[idx, 0] = 0.5
        
        # Process impact values (second column)
        impact_values = heatmap_data['Impact'].values  # Convert to numpy array
        impact_abs_max = np.abs(impact_values).max()
        if impact_abs_max > 0:
            z_values[:, 1] = impact_values / (2 * impact_abs_max) + 0.5
        else:
            z_values[:, 1] = 0.5
        
        # Create heatmap figure
        heatmap_fig = go.Figure()
        
        # Add heatmap trace
        heatmap_fig.add_trace(go.Heatmap(
            y=heatmap_data['Feature'],
            x=['Feature Value', 'Feature Impact'],
            z=z_values,
            colorscale='RdBu',
            text=np.column_stack([
                heatmap_data['Value'],
                heatmap_data['Impact'].round(4).astype(str)
            ]),
            texttemplate='%{text}',
            textfont={"size": 12},
            showscale=True
        ))
        
        # Update heatmap layout
        heatmap_fig.update_layout(
            title={
                'text': 'Feature Values and Impacts Heatmap',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            height=600,
            plot_bgcolor='white',
            yaxis=dict(
                automargin=True,
                title='Features',
                tickfont=dict(size=12)
            ),
            xaxis=dict(
                title='Analysis Type',
                tickfont=dict(size=12)
            ),
            margin=dict(l=200, r=50, t=100, b=50)
        )
        
        return waterfall_fig, heatmap_fig, shap_values_array
        
    except Exception as e:
        st.error(f"SHAP analysis error: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None

def analyze_prediction(input_data, prediction, model, X, y):
    """Analyze the prediction in context of training data"""
    # Create prediction vs actual distribution plot
    y_pred = model.predict(X)
    
    fig = go.Figure()
    
    # Add training data distribution
    fig.add_trace(go.Histogram(
        x=y,
        name='Training Data',
        opacity=0.7,
        nbinsx=30
    ))
    
    # Add current prediction
    fig.add_trace(go.Scatter(
        x=[prediction],
        y=[0],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        name='Current Prediction'
    ))
    
    fig.update_layout(
        title='Prediction in Context of Training Data Distribution',
        xaxis_title='CH3OH Selectivity(C%)',
        yaxis_title='Count',
        height=400
    )
    
    return fig

def make_prediction(model, input_data):
    """Make prediction using XGBoost model"""
    try:
        # Convert input data to numpy array if needed
        if isinstance(input_data, pd.DataFrame):
            input_array = input_data.values
        else:
            input_array = input_data
            
        # Make prediction directly using the model
        prediction = model.predict(input_array)
        
        # Return the first prediction if we have a single sample
        if isinstance(prediction, np.ndarray) and len(prediction) == 1:
            return prediction[0]
        return prediction
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error("Detailed input shape: " + str(input_data.shape))
        st.error("Input data types: " + str(input_data.dtypes))
        return None

def main():
    st.title("🧪 Methanol Selectivity Prediction System")
    st.write("Advanced Machine Learning-based Prediction and Analysis System")
    
    # Load model and data
    model, scaler, X, y, feature_names = load_model_and_data()
    if model is None:
        return
    
    # Create tabs
    tabs = st.tabs([
        "Parameter Input",
        "Prediction Analysis",
        "Model Insights",
        "Prediction History"
    ])
    
    # Parameter Input tab
    with tabs[0]:
        st.header("Parameter Settings")
        
        col1, col2 = st.columns(2)
        input_data = {}
        
        with col1:
            st.subheader("Reaction Conditions")
            for feature in ['Temp(℃)', 'Pressure (Mpa)', 'GHSV (h-1)']:
                input_data[feature] = st.slider(
                    feature,
                    min_value=FEATURES[feature]['min'],
                    max_value=FEATURES[feature]['max'],
                    value=FEATURES[feature]['default']
                )
        
        with col2:
            st.subheader("Catalyst Parameters")
            for feature in ['Cu(wt%)', 'Zn(wt%)']:
                input_data[feature] = st.slider(
                    feature,
                    min_value=FEATURES[feature]['min'],
                    max_value=FEATURES[feature]['max'],
                    value=FEATURES[feature]['default']
                )
        
        with st.expander("Advanced Parameters"):
            col3, col4 = st.columns(2)
            remaining_features = [f for f in FEATURES.keys() 
                               if f not in ['Temp(℃)', 'Pressure (Mpa)', 'GHSV (h-1)', 'Cu(wt%)', 'Zn(wt%)']]
            
            for i, feature in enumerate(remaining_features):
                with col3 if i < len(remaining_features)//2 else col4:
                    if 'options' in FEATURES[feature]:
                        input_data[feature] = st.selectbox(
                            feature,
                            options=FEATURES[feature]['options'],
                            index=FEATURES[feature]['options'].index(FEATURES[feature]['default'])
                        )
                    else:
                        input_data[feature] = st.slider(
                            feature,
                            min_value=FEATURES[feature]['min'],
                            max_value=FEATURES[feature]['max'],
                            value=FEATURES[feature]['default']
                        )
    
    # Make prediction when button is clicked
    if st.button("Make Prediction", type="primary"):
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical features
            input_df = encode_categorical_features(input_df)
            
            # Make prediction
            prediction = make_prediction(model, input_df)
            
            if prediction is not None:
                # Display immediate feedback
                st.success(f"Prediction successful! Selectivity: {prediction:.2f}%")
                
                # Store prediction in session state
                if 'current_prediction' not in st.session_state:
                    st.session_state.current_prediction = {}
                
                st.session_state.current_prediction = {
                    'input_data': input_data,
                    'prediction': prediction,
                    'timestamp': datetime.now()
                }
                
                # Store in prediction history
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                
                st.session_state.prediction_history.append(st.session_state.current_prediction)
                
                # Switch to Prediction Analysis tab
                tabs[1].active = True
            else:
                st.error("Failed to make prediction. Please check your input values.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check if all input parameters are valid.")
    
    # Prediction Analysis tab
    with tabs[1]:
        if 'current_prediction' in st.session_state:
            pred = st.session_state.current_prediction['prediction']
            input_data = st.session_state.current_prediction['input_data']
            
            # Display prediction
            st.header("Prediction Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Selectivity", f"{pred:.2f}%")
            
            try:
                # Create input DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Create SHAP analysis
                waterfall_fig, heatmap_fig, shap_values = create_shap_analysis(
                    model, input_df, list(input_data.keys())
                )
                
                if waterfall_fig is not None and heatmap_fig is not None:
                    # Display SHAP analysis
                    st.subheader("Feature Impact Analysis")
                    
                    # Display heatmap
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    # Display waterfall plot
                    st.plotly_chart(waterfall_fig, use_container_width=True)
                    
                    # Display feature contributions table
                    st.subheader("Feature Contributions")
                    contrib_df = pd.DataFrame({
                        'Feature': list(input_data.keys()),
                        'Value': list(input_data.values()),
                        'Impact': shap_values,
                        'Absolute Impact': np.abs(shap_values)
                    }).sort_values('Absolute Impact', ascending=False)
                    
                    # Format the table
                    st.dataframe(
                        contrib_df.style.format({
                            'Impact': '{:.4f}',
                            'Absolute Impact': '{:.4f}'
                        }).background_gradient(
                            subset=['Impact'],
                            cmap='RdBu',
                            vmin=-contrib_df['Absolute Impact'].max(),
                            vmax=contrib_df['Absolute Impact'].max()
                        )
                    )
                
                # Create distribution plot
                dist_fig = analyze_prediction(input_df, pred, model, X, y)
                if dist_fig is not None:
                    st.subheader("Prediction Distribution")
                    st.plotly_chart(dist_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in prediction analysis: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
    
    # Model Insights tab
    with tabs[2]:
        st.header("Model Insights")
        
        # Feature importance plot
        importance_fig = create_feature_importance_plot(model, list(FEATURES.keys()))
        st.plotly_chart(importance_fig, use_container_width=True)
        
        # Model performance metrics
        st.subheader("Model Performance")
        y_pred = model.predict(X)
        r2 = np.corrcoef(y, y_pred)[0,1]**2
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        mae = np.mean(np.abs(y - y_pred))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("MAE", f"{mae:.4f}")
    
    # Prediction History tab
    with tabs[3]:
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            st.header("Prediction History")
            
            # Convert history to DataFrame
            history_df = pd.DataFrame([
                {
                    'Timestamp': pred['timestamp'],
                    'Prediction': pred['prediction'],
                    **pred['input_data']
                }
                for pred in st.session_state.prediction_history
            ])
            
            # Display history
            st.dataframe(history_df)
            
            # Download button
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download Prediction History",
                data=csv,
                file_name='prediction_history.csv',
                mime='text/csv'
            )
        else:
            st.info("No prediction history available yet.")

if __name__ == "__main__":
    main() 