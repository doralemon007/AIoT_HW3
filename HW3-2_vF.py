import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

# Set up Streamlit layout
st.title("SVM Classifier with Dataset Selection and 3D Visualization")

# Sidebar parameter for dataset type selection
dataset_type = st.sidebar.selectbox("Choose Dataset Type", ["Circular (2D)", "Simple 1D"])

# Define parameters and generate the dataset based on selection
if dataset_type == "Circular (2D)":
    # Parameters for circular dataset
    st.sidebar.header("2D Circular Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=1000, value=300, step=50)
    factor = st.sidebar.slider("Factor (Circle Separation)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    noise = st.sidebar.slider("Noise Level", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
    
    # Generate circular dataset
    X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=42)
    plot_type = "3D"
    
elif dataset_type == "Simple 1D":
    # Parameters for 1D dataset
    st.sidebar.header("1D Simple Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=1000, value=300, step=50)
    a = st.sidebar.slider("Lower Bound (a)", min_value=-10.0, max_value=0.0, value=-3.0, step=0.1)
    b = st.sidebar.slider("Upper Bound (b)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    noise = st.sidebar.slider("Noise Level", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # Generate 1D dataset
    x = np.linspace(-10, 10, n_samples)
    y = np.where((x < a) | (x > b), 1, 0)
    x += np.random.normal(0, noise, x.shape)  # Adding noise
    X = x.reshape(-1, 1)  # Reshape for compatibility with sklearn
    plot_type = "2D"

# Sidebar parameters for SVM model
st.sidebar.header("SVM Parameters")
kernel = st.sidebar.selectbox("Kernel Type", ["rbf", "linear", "poly"])
gamma = st.sidebar.selectbox("Gamma (for RBF kernel)", ["scale", "auto"])
C = st.sidebar.slider("Regularization (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# Train SVM model
svm_model = SVC(kernel=kernel, gamma=gamma, C=C)
svm_model.fit(X, y)
y_pred = svm_model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Display model accuracy
st.write(f"Model Accuracy: {accuracy:.2f}")

# Visualization based on dataset type
if plot_type == "3D":
    # Prepare data for 3D plotting of decision boundary
    x_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    y_range = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    xy_mesh = np.c_[xx.ravel(), yy.ravel()]
    z = svm_model.decision_function(xy_mesh).reshape(xx.shape)

    # Plot in 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the classes
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], 0, color='blue', label='Class 0', alpha=0.6)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], 0, color='orange', label='Class 1', alpha=0.6)

    # Decision boundary surface
    ax.plot_surface(xx, yy, z, rstride=1, cstride=1, color='green', alpha=0.3, edgecolor='none')

    # Axis labels and legend
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Decision Function")
    ax.set_title("3D Visualization of SVM Decision Boundary on Circular Data")
    ax.legend()

    # Display 3D plot in Streamlit
    st.pyplot(fig)

elif plot_type == "2D":
    # Plot in 2D
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Data points')
    ax.plot(X, svm_model.decision_function(X), color='red', linestyle='--', label='SVM decision function')

    # Display details
    ax.set_xlabel("Feature (x)")
    ax.set_ylabel("Decision Function Value")
    ax.set_title("2D Visualization of SVM Decision Boundary on 1D Simple Data")
    ax.legend()

    # Display 2D plot in Streamlit
    st.pyplot(fig)
