import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.metrics import accuracy_score

# Custom function for generating spiral dataset
def generate_spiral(n_samples, revolutions, noise=0.1):
    theta = np.sqrt(np.random.rand(n_samples)) * revolutions * 2 * np.pi
    r = theta * (1 + noise * np.random.rand(n_samples))
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    X = np.vstack((np.hstack((x1, -x1)), np.hstack((x2, -x2)))).T
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
    return X, y

# Set up Streamlit layout
st.title("SVM Classifier with Multiple Datasets and Visualization Options")

# Sidebar for dataset type selection
dataset_type = st.sidebar.selectbox(
    "Choose Dataset Type", ["Moons", "Spiral", "Elliptical Blobs", "Circular (2D)", "Simple 1D"]
)

# Dataset generation based on selection
if dataset_type == "Circular (2D)":
    st.sidebar.header("2D Circular Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, 50)
    factor = st.sidebar.slider("Factor (Circle Separation)", 0.1, 1.0, 0.5, 0.1)
    noise = st.sidebar.slider("Noise Level", 0.0, 0.3, 0.05, 0.01)
    X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=42)
    plot_type = "3D"

elif dataset_type == "Simple 1D":
    st.sidebar.header("1D Simple Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, 50)
    a = st.sidebar.slider("Lower Bound (a)", -10.0, 0.0, -3.0, 0.1)
    b = st.sidebar.slider("Upper Bound (b)", 0.0, 10.0, 3.0, 0.1)
    noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.5, 0.1)
    x = np.linspace(-10, 10, n_samples)
    y = np.where((x < a) | (x > b), 1, 0)
    x += np.random.normal(0, noise, x.shape)  # Adding noise
    X = x.reshape(-1, 1)
    plot_type = "1D"

elif dataset_type == "Moons":
    st.sidebar.header("Moons Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, 50)
    noise = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.2, 0.01)
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    plot_type = "2D"

elif dataset_type == "Spiral":
    st.sidebar.header("Spiral Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples per Class", 100, 500, 150, 50)
    revolutions = st.sidebar.slider("Number of Revolutions", 1, 5, 2, 1)
    noise = st.sidebar.slider("Noise Level", 0.0, 0.3, 0.1, 0.01)
    X, y = generate_spiral(n_samples=n_samples, revolutions=revolutions, noise=noise)
    plot_type = "2D"

elif dataset_type == "Elliptical Blobs":
    st.sidebar.header("Elliptical Blobs Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, 50)
    cluster_std_1 = st.sidebar.slider("Cluster 1 Std Dev", 0.5, 3.0, 1.5, 0.1)
    cluster_std_2 = st.sidebar.slider("Cluster 2 Std Dev", 0.5, 3.0, 2.5, 0.1)
    X, y = make_blobs(n_samples=n_samples, centers=[(-3, -3), (3, 3)], 
                      cluster_std=[cluster_std_1, cluster_std_2], random_state=42)
    plot_type = "2D"

# Sidebar for SVM parameters
st.sidebar.header("SVM Parameters")
kernel = st.sidebar.selectbox("Kernel Type", ["rbf", "linear", "poly"])
gamma = st.sidebar.selectbox("Gamma (for RBF kernel)", ["scale", "auto"])
C = st.sidebar.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)

# Train the SVM model
svm_model = SVC(kernel=kernel, gamma=gamma, C=C)
svm_model.fit(X, y)
y_pred = svm_model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Display model accuracy
st.write(f"Model Accuracy: {accuracy:.2f}")

# Visualization based on dataset type
if plot_type == "3D":
    x_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    y_range = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    xy_mesh = np.c_[xx.ravel(), yy.ravel()]
    z = svm_model.decision_function(xy_mesh).reshape(xx.shape)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], 0, color='blue', alpha=0.6, label='Class 0')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], 0, color='orange', alpha=0.6, label='Class 1')
    ax.plot_surface(xx, yy, z, rstride=1, cstride=1, color='green', alpha=0.3, edgecolor='none')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Decision Function")
    ax.set_title("3D Visualization of SVM Decision Boundary on Circular Data")
    ax.legend()
    st.pyplot(fig)

elif plot_type == "2D":
    # For 2D datasets, we show contours of the decision function
    x_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    y_range = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    xy_mesh = np.c_[xx.ravel(), yy.ravel()]
    z = svm_model.decision_function(xy_mesh).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0', edgecolor='k')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='orange', label='Class 1', edgecolor='k')
    ax.contourf(xx, yy, z, levels=[-1, 0, 1], alpha=0.2, colors=['blue', 'orange'])
    ax.contour(xx, yy, z, levels=[0], linewidths=2, colors='black')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{dataset_type} Dataset with SVM Decision Boundary")
    ax.legend()
    st.pyplot(fig)

elif plot_type == "1D":
    # For 1D dataset, plot decision boundary as a line
    fig, ax = plt.subplots(figsize=(10, 6))
    # Scatter plot for class labels
    ax.scatter(X[y == 0], y[y == 0], color='blue', label='Class 0', edgecolor='k')
    ax.scatter(X[y == 1], y[y == 1], color='orange', label='Class 1', edgecolor='k')

    # Plot the decision boundary
    x_range = np.linspace(X.min() - 1, X.max() + 1, 500).reshape(-1, 1)
    decision_boundary = svm_model.decision_function(x_range)
    ax.plot(x_range, decision_boundary, color='red', linestyle='--', label='Decision Boundary')
    
    # Highlight the threshold for classification (decision boundary at 0)
    ax.axhline(0, color='black', linestyle=':', linewidth=1)
    
    # Set labels, title, and legend
    ax.set_xlabel("Feature (x)")
    ax.set_ylabel("Decision Function Value")
    ax.set_title("1D Simple Dataset with SVM Decision Boundary")
    ax.legend()
    st.pyplot(fig)