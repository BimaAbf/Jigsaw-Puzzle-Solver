# 🧩 Jigsaw Puzzle Solver

An intelligent computer vision application that automatically solves square jigsaw puzzles. This project uses advanced algorithms to analyze puzzle pieces, determine their compatibility, and reconstruct the original image.

## ✨ Features

- **Automatic Grid Detection**: Analyzes the input image to determine the puzzle grid size (e.g., 2x2, 3x3, 4x4, etc.).
- **Intelligent Segmentation**: Automatically slices the input image into puzzle pieces based on the detected grid.
- **Multiple Solving Strategies**:
  - **Backtracking**: Exact solver for small puzzles (2x2, 3x3).
  - **Beam Search**: Heuristic search for medium puzzles (4x4, 5x5).
  - **Forest Solver**: Advanced Kruskal's algorithm + Backtracking for larger puzzles.
- **Interactive UI**: User-friendly web interface built with Streamlit.
- **Visual Feedback**: Displays the input puzzle, segmented pieces, and the final solved result.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Jigsaw-Puzzle-Solver
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, install the following packages: `streamlit`, `opencv-python`, `numpy`, `matplotlib`)*

## 🚀 Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2.  **Use the Web Interface**:
    - Upload a puzzle image (JPG or PNG).
    - The app will automatically analyze the image and detect the grid size.
    - Click **"Solve Puzzle"** to start the process.
    - View the segmented pieces and the final reconstructed image.

## 🧠 How It Works

1.  **Preprocessing**: The input image is analyzed to detect the number of pieces (grid size).
2.  **Segmentation**: The image is sliced into individual pieces.
3.  **Feature Extraction**: The borders of each piece are extracted.
4.  **Compatibility Analysis**: A dissimilarity matrix is computed using the Normalized Sum of Squared Differences (NSSD) between piece borders.
5.  **Solving**:
    - **Small Puzzles**: Uses a backtracking algorithm to find the exact solution.
    - **Medium Puzzles**: Uses Beam Search to explore the most promising partial solutions.
    - **Large Puzzles**: Uses a "Forest" approach, building small clusters of matching pieces (trees) and merging them.

## 📂 Project Structure

```
Jigsaw-Puzzle-Solver/
├── app.py                  # Streamlit application entry point
├── solver.py               # Core solver logic and pipeline
├── solver_forest.py        # Advanced solver for large puzzles
├── jigsaw/                 # Package for image analysis and extraction
│   ├── extractor.py        # Grid size detection logic
│   └── ...
├── evaluation/             # Evaluation metrics and datasets
├── thresholding/           # Thresholding analysis tools
└── README.md               # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.