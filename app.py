import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import random
from solver import (
    run_pipeline,
    get_image_descriptor,
    slice_image,
    solve_puzzle_lab2
)

# Set page configuration
st.set_page_config(
    page_title="Jigsaw Puzzle Solver",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        color: white;
    }
    .success-box {
        padding: 1rem;
        background-color: #D4EDDA;
        color: #155724;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🧩 Jigsaw Solver")
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Upload a puzzle image 📸
    2. We analyze the grid structure 🔍
    3. Our AI solves it! 🧠
    """)
    st.markdown("---")
    st.info("Supported formats: JPG, PNG")

# Main Content
st.title("🖼️ Jigsaw Puzzle Solver")
st.markdown("### Upload a scrambled puzzle image to solve it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temporary directory
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.close()
    image_path = tfile.name

    # Display original image
    original_image = cv2.imread(image_path)
    if original_image is not None:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Puzzle")
            st.image(original_image, use_container_width=True, channels="RGB")
            
        with col2:
            st.subheader("🧩 Segmented Pieces")
            pieces_placeholder = st.empty()
            pieces_placeholder.info("Click 'Solve Puzzle' to see pieces here.")

        # Analysis Section
        st.markdown("---")
        st.subheader("🔍 Analysis & Solving")
        
        if st.button("🚀 Solve Puzzle", key="solve_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Analysis
                status_text.text("Analyzing grid structure...")
                progress_bar.progress(10)
                
                descriptor = get_image_descriptor(image_path)
                if not descriptor:
                    st.error("Failed to analyze image.")
                    st.stop()
                    
                grid_size = descriptor['grid_size']
                st.success(f"Detected Grid Size: {grid_size}x{grid_size}")
                
                # Step 2: Segmentation
                status_text.text("Segmenting image...")
                progress_bar.progress(30)
                
                pieces = slice_image(original_image, grid_size)
                
                with pieces_placeholder.container():
                    # Display pieces in a grid
                    for r in range(grid_size):
                        cols = st.columns(grid_size)
                        for c in range(grid_size):
                            idx = r * grid_size + c
                            if idx < len(pieces):
                                cols[c].image(pieces[idx], use_container_width=True)
                            
                # Step 3: Solving
                status_text.text("Solving puzzle...")
                progress_bar.progress(60)
                
                solved_canvas = solve_puzzle_lab2(image_path, descriptor, shuffle=False, show_plot=False)
                
                progress_bar.progress(100)
                status_text.text("Done!")
                
                if solved_canvas is not None:
                    st.markdown("---")
                    st.subheader("✨ Solved Result")
                    
                    # Center the result
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        st.image(solved_canvas, use_container_width=True, channels="RGB")
                        st.balloons()
                else:
                    st.error("Could not find a solution.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                # Cleanup temp file
                try:
                    os.unlink(image_path)
                except:
                    pass
    else:
        st.error("Could not load image. Please try another file.")
