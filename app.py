import streamlit as st
import yfinance as yf
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
st.title("Time Series Forecasting with Deep Learning")
st.write("""
Bu uygulama Yahoo Finance verilerini kullanarak hisse senedi zaman serisi analizi yapar.
Kapanış fiyatı, 20 günlük ve 50 günlük hareketli ortalamalar grafik üzerinde gösterilir.
Bir sonraki aşamada derin öğrenme modeli ile gelecek fiyat tahmini yapılacaktır.
""")

hisseler = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "THYAO.IS",
    "ASELS.IS",
    "TUPRS.IS",
    "EREGL.IS",
    "BIMAS.IS"
]

ticker = st.selectbox(
    "Hisse Seçiniz",
    hisseler
)
period = st.selectbox(
    "Veri Aralığı",
    ["1y", "3y", "5y", "max"]
)
data = yf.download(ticker, period=period, auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

st.subheader("Son Veriler")
st.write(data.tail())

chart_data = pd.DataFrame()
chart_data["Close"] = data["Close"]
chart_data["MA20"] = data["Close"].rolling(20).mean()
chart_data["MA50"] = data["Close"].rolling(50).mean()

st.subheader("Fiyat Grafiği")
st.line_chart(chart_data)
en_yuksek = data["High"].max()
en_dusuk = data["Low"].min()
ortalama = data["Close"].mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("En Yüksek", f"{en_yuksek:.2f}")

with col2:
    st.metric("En Düşük", f"{en_dusuk:.2f}")

with col3:
    st.metric("Ortalama", f"{ortalama:.2f}")
son_fiyat = data["Close"].iloc[-1]

st.metric(
    label="Son Kapanış Fiyatı",
    value=f"${son_fiyat:.2f}"
)

st.subheader("Yapay Zeka Fiyat Tahmini")

data["Gun"] = np.arange(len(data))

X = data[["Gun"]]
y = data["Close"]

model = LinearRegression()
model.fit(X, y)

gelecek_gun = len(data) + 30

tahmin = model.predict([[gelecek_gun]])[0]

st.metric(
    label="30 Gün Sonraki Tahmini Fiyat",
    value=f"${tahmin:.2f}"
)
# LSTM Derin Öğrenme Tahmini

kapanis = data["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(kapanis)

X_lstm = []
y_lstm = []

for i in range(60, len(scaled_data)):
    X_lstm.append(scaled_data[i-60:i, 0])
    y_lstm.append(scaled_data[i, 0])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

X_lstm = np.reshape(
    X_lstm,
    (X_lstm.shape[0], X_lstm.shape[1], 1)
)

model_lstm = Sequential()

model_lstm.add(
    LSTM(
        units=50,
        return_sequences=False,
        input_shape=(X_lstm.shape[1], 1)
    )
)

model_lstm.add(Dense(1))

model_lstm.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model_lstm.fit(
    X_lstm,
    y_lstm,
    epochs=3,
    batch_size=32,
    verbose=0
)

son_60 = scaled_data[-60:]
son_60 = np.reshape(
    son_60,
    (1, 60, 1)
)

tahmin_lstm = model_lstm.predict(
    son_60,
    verbose=0
)

tahmin_lstm = scaler.inverse_transform(
    tahmin_lstm
)

st.metric(
    "LSTM 30 Gün Sonrası Tahmin",
    f"${tahmin_lstm[0][0]:.2f}"
)
y_pred = model.predict(X)

rmse = np.sqrt(
    mean_squared_error(y, y_pred)
)

st.metric(
    "Model Hata Oranı (RMSE)",
    f"{rmse:.2f}"
)
st.subheader("LSTM Tahmin Grafiği")

lstm_grafik = pd.DataFrame({
    "Gerçek Son Fiyat": [son_fiyat],
    "LSTM Tahmin": [tahmin_lstm[0][0]]
})

st.bar_chart(lstm_grafik)
st.subheader("Gerçek ve Tahmin Karşılaştırması")

karsilastirma = pd.DataFrame({
    "Gerçek": y.tail(30).values,
    "Tahmin": y_pred[-30:]
})

st.line_chart(karsilastirma)
st.subheader("Model Karşılaştırması")

karsi1, karsi2 = st.columns(2)

with karsi1:
    st.metric(
        "Linear Regression",
        f"${tahmin:.2f}"
    )

with karsi2:
    st.metric(
        "LSTM",
        f"${tahmin_lstm[0][0]:.2f}"
    )
    st.subheader("Model Değerlendirmesi")

if abs(tahmin - son_fiyat) < abs(tahmin_lstm[0][0] - son_fiyat):
    st.success("En başarılı model: Linear Regression")
else:
    st.success("En başarılı model: LSTM")
