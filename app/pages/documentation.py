import streamlit as st

def render_documentation():
    st.set_page_config(page_title="Documentation", page_icon="📚")
    
    st.title("📚 System Documentation")
    
    st.markdown("""
    This documentation provides an overview of the key
    components and classes in the AutoML system.
    """)

    # Core System Components
    st.header("🔧 Core System Components")
    
    with st.expander("AutoMLSystem"):
        st.markdown("""
        The `AutoMLSystem` is the main system class implementing
        a singleton pattern to ensure only one instance exists.
        
        **Key Features:**
        - Manages the artifact registry
        - Handles storage and database operations
        - Provides centralized access to system components
        """)

    with st.expander("ArtifactRegistry"):
        st.markdown("""
        The `ArtifactRegistry` manages registration
        and listing of artifacts in the system.
        
        **Key Operations:**
        - Register new artifacts
        - List existing artifacts
        - Get specific artifacts
        - Delete artifacts
        """)

    # Machine Learning Components
    st.header("🤖 Machine Learning Components")
    
    with st.expander("Pipeline"):
        st.markdown("""
        The `Pipeline` class executes machine learning workflows.
        
        **Key Features:**
        - Model training and evaluation
        - Data preprocessing
        - Feature validation
        - Metrics computation
        """)

    with st.expander("Models"):
        st.markdown("""
        **Available Models:**
        1. Classification Models:
           - Tree Classification
           - K-Nearest Neighbors
           - Stochastic Gradient
           - Text Classification
        
        2. Regression Models:
           - Multiple Linear Regression
           - Ridge Regression
           - Support Vector Regression
        """)

    with st.expander("Metrics"):
        st.markdown("""
        **Available Metrics:**
        
        Classification Metrics:
        - Accuracy
        - F1 Score
        - Specificity
        
        Regression Metrics:
        - Mean Squared Error
        - Mean Absolute Error
        - R2 Score
        """)

    # Data Components
    st.header("📊 Data Components")
    
    with st.expander("Dataset"):
        st.markdown("""
        The `Dataset` class handles data storage and manipulation.
        
        **Features:**
        - Data loading and saving
        - Conversion to/from pandas DataFrames
        - Metadata management
        """)

    with st.expander("Feature"):
        st.markdown("""
        The `Feature` class represents individual data features.
        
        **Supported Types:**
        - Categorical
        - Numerical
        - Text
        - Image
        - Audio
        - Video
        """)

    # Storage Components
    st.header("💾 Storage Components")
    
    with st.expander("Storage"):
        st.markdown("""
        The storage system handles data persistence.
        
        **Components:**
        - LocalStorage: File-based storage implementation
        - Database: JSON-based metadata storage
        """)

    # User Interface Components
    st.header("🖥️ User Interface Components")
    
    with st.expander("Available Pages"):
        st.markdown("""
        1. **Datasets Page:**
           - Upload new datasets
           - View existing datasets
           - Remove datasets
        
        2. **Modelling Page:**
           - Select models and features
           - Configure hyperparameters
           - Train and evaluate models
        
        3. **Deployment Page:**
           - Load trained pipelines
           - Make predictions
           - Evaluate model performance
        """)

if __name__ == "__main__":
    render_documentation()