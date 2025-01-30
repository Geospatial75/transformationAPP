import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def estimate_transformation(nad27_coords, nad83_coords):
    """
    Estimates the 7 transformation parameters (3 translations, 3 rotations, 1 scale factor)
    from NAD27 to NAD83 using least squares adjustment.
    """
    n = len(nad27_coords)
    
    # Compute misclosure vector w (differences in coordinates)
    w = pd.DataFrame(nad83_coords).values - pd.DataFrame(nad27_coords).values
    w = w.flatten()
    
    # Construct design matrix A
    A = []
    for x, y, z in nad27_coords:
        A.append([1, 0, 0, x, y, z, x])
        A.append([0, 1, 0, -z, x, y, y])
        A.append([0, 0, 1, y, -x, z, z])
    
    A = pd.DataFrame(A).values
    
    # Least squares solution: (A^T A)^{-1} A^T w
    AtA = pd.DataFrame(A).T.dot(pd.DataFrame(A))
    AtW = pd.DataFrame(A).T.dot(pd.DataFrame(w))
    delta = pd.DataFrame(AtA).invert().dot(pd.DataFrame(AtW)).values.flatten()
    
    # Extract parameters
    t_x, t_y, t_z, eps_x, eps_y, eps_z, k = delta
    
    return {
        'Translation (t_x, t_y, t_z)': (t_x, t_y, t_z),
        'Rotation (eps_x, eps_y, eps_z)': (eps_x, eps_y, eps_z),
        'Scale factor (k)': k
    }

# Streamlit UI
st.title("NAD27 to NAD83 Transformation App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Assuming CSV has columns: 'NAD27_X', 'NAD27_Y', 'NAD27_Z', 'NAD83_X', 'NAD83_Y', 'NAD83_Z'
    nad27_coords = data[['NAD27_X', 'NAD27_Y', 'NAD27_Z']].values.tolist()
    nad83_coords = data[['NAD83_X', 'NAD83_Y', 'NAD83_Z']].values.tolist()
    
    # Estimate transformation parameters
    parameters = estimate_transformation(nad27_coords, nad83_coords)
    
    # Display results
    st.write("### Transformation Parameters")
    for key, value in parameters.items():
        st.write(f"**{key}:** {value}")
    
    # Save results to CSV
    output_file = "transformation_results.csv"
    pd.DataFrame([parameters]).to_csv(output_file, index=False)
    
    # Provide download link for results
    with open(output_file, "rb") as file:
        st.download_button(label="Download Transformation Results", data=file, file_name=output_file, mime="text/csv")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    nad27_x, nad27_y = zip(*[(x, y) for x, y, _ in nad27_coords])
    nad83_x, nad83_y = zip(*[(x, y) for x, y, _ in nad83_coords])
    
    ax.scatter(nad27_x, nad27_y, color='blue', label='NAD27 Points')
    ax.scatter(nad83_x, nad83_y, color='red', label='NAD83 Points')
    
    for i in range(len(nad27_x)):
        ax.plot([nad27_x[i], nad83_x[i]], [nad27_y[i], nad83_y[i]], 'k--')
    
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("2D Transformation Visualization: NAD27 to NAD83")
    ax.legend()
    ax.grid()
    
    st.pyplot(fig)
