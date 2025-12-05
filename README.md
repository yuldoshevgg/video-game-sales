# Video Game Sales Analysis

A complete **machine learning project** analyzing **64,016 video game titles** (1971–2024) with features such as critic scores, genres, console types, regional sales, publishers, and more.

The project includes:
* **Exploratory Data Analysis (EDA)**
* **Data Preprocessing and Feature Engineering**
* **Machine Learning Model Training/Evaluation**
* **Deployment** using a multi-page **Streamlit Web App**

---

## 📦 Installation

To get a local copy up and running, follow these simple steps.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yuldoshevgg/video-game-sales.git
    cd video-game-sales
    ```

2.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Mac/Linux
    venv\Scripts\activate     # Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## 📓 Usage — Jupyter Notebook

The main development and analysis work is documented in the Jupyter Notebook.

1.  **Launch Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2.  **Open the file:** `01_ml_coursework.ipynb`

This notebook covers:
* Data loading
* EDA & visualization
* Preprocessing
* Model training
* Model evaluation

---

## 🌐 Running the Streamlit App

Run the interactive multi-page web application:

```bash
streamlit run app/streamlit_app.py