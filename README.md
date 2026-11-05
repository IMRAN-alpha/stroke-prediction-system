# Stroke Prediction — EDA Live Demo

This is a Streamlit dashboard for demonstrating dataset exploration and data visualization using the supplied healthcare stroke dataset. It is aligned with the documented student role: Data Visualization and EDA.

## Run on a laptop

1. Install Python 3.10 or newer.
2. Extract this ZIP into a folder.
3. Open Command Prompt/Terminal in that folder.
4. Install dependencies:

   `python -m pip install -r requirements.txt`

5. Start the dashboard:

   `python -m streamlit run app.py`

6. Open the local URL shown in the terminal (usually http://localhost:8501).

Keep `healthcare-dataset-stroke-data.csv` in the same folder as `app.py`.

## Scope and honesty

This package demonstrates EDA and visualizations. It does not perform live ML inference. The model metrics shown in the final tab are clearly labelled as values reported in the internship presentation. A separate prediction demo requires verified, loadable model artifacts and their original preprocessing pipeline.

This is an educational demonstration, not a medical diagnostic tool.
