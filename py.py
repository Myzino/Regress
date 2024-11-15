import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Set page config
st.set_page_config(page_title="Customer Analysis Dashboard", layout="wide")

# Function to create a bar plot for complaint frequency by 'Received Via'
def create_complaints_bar_plot(df):
    plt.figure(figsize=(10, 6))
    
    # Create binary column for complaints
    df['Complaint_Binary'] = df['Customer Complaint'].apply(lambda x: 1 if pd.notna(x) else 0)
    
    # Calculate statistics
    complaint_stats = df.groupby('Received Via')['Complaint_Binary'].agg(['mean', 'median']).reset_index()
    
    # Create bar plot
    x = np.arange(len(complaint_stats))
    width = 0.35
    
    plt.bar(x - width/2, complaint_stats['mean'], width, label='Mean', color='skyblue')
    plt.bar(x + width/2, complaint_stats['median'], width, label='Median', color='lightgreen')
    
    plt.xlabel('Received Via')
    plt.ylabel('Complaint Frequency')
    plt.title('Mean and Median Complaint Frequency by Received Via')
    plt.xticks(x, complaint_stats['Received Via'], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    return plt

def create_combined_mode_pie_chart(df):
    plt.figure(figsize=(12, 8))
    
    columns_of_interest = ['Customer Complaint', 'Received Via', 'City', 'Status']
    
    modes_data = {}
    for col in columns_of_interest:
        if col in df.columns:
            if col == 'Customer Complaint':
                mode_val = df[col].fillna('No Complaint').mode().iloc[0]
                mode_freq = df[col].fillna('No Complaint').value_counts().iloc[0]
            else:
                mode_val = df[col].mode().iloc[0]
                mode_freq = df[col].value_counts().iloc[0]
            modes_data[f"{col}\n({mode_val})"] = mode_freq
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(modes_data)))
    plt.pie(modes_data.values(), labels=modes_data.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Most Frequent Categories for Selected Variables')
    
    return plt

def create_dominance_bar_plots(df):
    columns = df.columns
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5*n_rows))
    fig.suptitle('Dominant Categories Analysis for All Columns', fontsize=16, y=1.02)
    
    axes_flat = axes.flatten()
    
    for idx, (col, ax) in enumerate(zip(columns, axes_flat)):
        if col == 'Customer Complaint':
            value_counts = df[col].fillna('No Complaint').value_counts()
        else:
            value_counts = df[col].value_counts()
        
        top_n = value_counts.head(5)
        percentages = (top_n / len(df) * 100).round(1)
        
        bars = ax.bar(range(len(top_n)), percentages, color='lightgray')
        bars[0].set_color('skyblue')
        
        for i, v in enumerate(percentages):
            ax.text(i, v, f'{v}%', ha='center', va='bottom')
        
        ax.set_title(f'{col}\nDominant: {top_n.index[0]}', pad=10)
        ax.set_xticks(range(len(top_n)))
        ax.set_xticklabels(top_n.index, rotation=45, ha='right')
        ax.set_ylabel('Percentage (%)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    for idx in range(len(columns), len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    plt.tight_layout()
    return fig

def perform_linear_regression(df):
    if 'Status' in df.columns:
        df['Status'] = df['Status'].apply(lambda x: 1 if x == 'Closed' else 0)
    
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_columns) < 2:
        st.warning("The dataset must have at least two numerical columns for regression.")
        return
    
    target_column = 'Status'
    feature_columns = [col for col in numerical_columns if col != target_column]
    
    X = df[feature_columns]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    st.write("### Regression Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R-squared Score", f"{r2:.4f}")
    with col2:
        st.metric("Mean Squared Error", f"{mse:.4f}")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(y_test, y_pred, alpha=0.6)
    ax1.plot([0, 1], [0, 1], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Actual vs Predicted Values')
    st.pyplot(fig1)
    
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    st.pyplot(fig2)

def main():
    st.title("Customer Analysis Dashboard")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success("Data loaded successfully!")
            
            # Show sample of the dataset
            st.write("### Sample Data")
            st.dataframe(df.head())
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["Complaint Analysis", "Mode Distribution", 
                                             "Category Dominance", "Regression Analysis"])
            
            with tab1:
                st.write("### Complaint Frequency Analysis")
                fig_complaints = create_complaints_bar_plot(df)
                st.pyplot(fig_complaints)
                plt.close()
            
            with tab2:
                st.write("### Mode Distribution for Key Variables")
                fig_pie = create_combined_mode_pie_chart(df)
                st.pyplot(fig_pie)
                plt.close()
            
            with tab3:
                st.write("### Category Dominance Analysis")
                fig_dominance = create_dominance_bar_plots(df)
                st.pyplot(fig_dominance)
                plt.close()
            
            with tab4:
                st.write("### Regression Analysis")
                perform_linear_regression(df)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure your CSV file contains the required columns: "
                    "'Customer Complaint', 'Received Via', 'City', 'Status'")
    
    else:
        st.info("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()