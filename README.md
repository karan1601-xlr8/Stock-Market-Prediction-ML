# 📈 Stock Market Prediction Web App

A deep learning-based web application built using **Streamlit** that predicts stock prices using historical data and a trained **Keras LSTM model**.
This project demonstrates how to fetch real-time stock data, preprocess it, visualize trends, and make predictions using machine learning.

---

## 🚀 Features

* 📊 Fetches real-time stock data using `yfinance`
* 🧠 Uses a trained **LSTM** model for stock price prediction
* 📉 Displays trend visualizations including:

  * Moving Averages (MA50, MA100, MA200)
  * Actual vs Predicted Prices
* 🔧 Interactive Streamlit interface for entering stock symbols
* 💾 Includes a Jupyter Notebook for training the prediction model

---

## 🧩 Project Structure

```
📦 Stock-Market-Prediction
├── app.py                              # Streamlit web application
├── Stock_Market_Prediction_Model_Creation.ipynb  # Model training notebook
├── Stock Predictions Model.keras        # Trained LSTM model (to be added)
└── README.md                            # Project documentation
```

---

## 🧠 Model Overview

The model is built using **Keras (TensorFlow backend)** and trained on historical stock closing prices.
It uses an **LSTM (Long Short-Term Memory)** architecture to capture time-series dependencies.

Key Steps in the Model:

1. Data preprocessing with `MinMaxScaler`
2. Sequence generation with a sliding window of 100 days
3. Model training and saving (`Stock Predictions Model.keras`)
4. Prediction and inverse scaling for interpretability

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Stock-Market-Prediction.git
cd Stock-Market-Prediction
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install streamlit yfinance numpy pandas keras matplotlib scikit-learn
```

### 3. Add the Model File

Place your trained model file in the same directory as `app.py`:

```
Stock Predictions Model.keras
```

### 4. Run the App

```bash
streamlit run app.py
```

Then open the local URL displayed in your terminal (usually `http://localhost:8501`).

---

## 🧾 Usage

1. Run the Streamlit app.
2. Enter a valid **stock ticker symbol** (e.g., `GOOG`, `AAPL`, `MSFT`).
3. The app will:

   * Download historical data (2012–2022)
   * Display price trend charts and moving averages
   * Predict future prices and visualize comparison graphs

---

## 📸 Example Output

**Price vs MA50**

```
A graph showing short-term moving average (red) vs. actual prices (green)
```

**Price vs MA100 vs MA200**

```
Displays long-term and medium-term trends to observe stock momentum
```

**Original vs Predicted Prices**

```
Compares LSTM-predicted prices with actual test data for evaluation
```

---

## 🧰 Technologies Used

| Library                | Purpose                                 |
| ---------------------- | --------------------------------------- |
| **Streamlit**          | Web interface for real-time interaction |
| **YFinance**           | Fetching live stock market data         |
| **Keras / TensorFlow** | Deep learning model (LSTM)              |
| **Pandas & NumPy**     | Data manipulation and preprocessing     |
| **Matplotlib**         | Visualization and plotting              |
| **Scikit-learn**       | Feature scaling and preprocessing       |

---

## 📊 Model Training (Notebook)

The included notebook `Stock_Market_Prediction_Model_Creation.ipynb` handles:

* Data collection and cleaning
* Feature scaling and sequence creation
* Model training and saving
* Performance evaluation

Once trained, the model file `Stock Predictions Model.keras` can be loaded directly in `app.py`.

---

## 🧑‍💻 Author

**Karan Raj**
Engineer | Backend Developer | Machine Learning Enthusiast
📍 NIT Agartala

---

## 🪪 License

This project is open-source under the **MIT License**.

---

## ⭐ Future Enhancements

* Add real-time prediction updates
* Integrate with live dashboards (Plotly / Altair)
* Support for multiple stock comparisons
* Include volume and sentiment analysis for better accuracy

---

## 🧠 Acknowledgments

* [Yahoo Finance API](https://pypi.org/project/yfinance/)
* [Streamlit Docs](https://docs.streamlit.io/)
* [Keras LSTM Tutorial](https://keras.io/examples/timeseries/)

