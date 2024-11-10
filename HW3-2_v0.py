import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

# Set up Streamlit layout
st.title("2D SVM Classifier with Adjustable Parameters and 3D Visualization")

# Sidebar parameters for dataset
st.sidebar.header("Dataset Parameters")
n_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=1000, value=300, step=50)
factor = st.sidebar.slider("Factor (Circle Separation)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
noise = st.sidebar.slider("Noise Level", min_value=0.0, max_value=0.3, value=0.05, step=0.01)

# Sidebar parameters for SVM model
st.sidebar.header("SVM Parameters")
kernel = st.sidebar.selectbox("Kernel Type", ["rbf", "linear", "poly"])
gamma = st.sidebar.selectbox("Gamma (for RBF kernel)", ["scale", "auto"])
C = st.sidebar.slider("Regularization (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# Generate circular dataset based on parameters
X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=42)

# Train SVM model
svm_model = SVC(kernel=kernel, gamma=gamma, C=C)
svm_model.fit(X, y)
y_pred = svm_model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Display dataset and model accuracy
st.write(f"Model Accuracy: {accuracy:.2f}")

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
