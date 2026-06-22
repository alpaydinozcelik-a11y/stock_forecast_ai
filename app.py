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

if data.empty:
    st.error("Yahoo Finance verisi alınamadı.")
    st.stop()

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

y_pred = model.predict(X)

rmse = np.sqrt(
    mean_squared_error(y, y_pred)
)







karsilastirma = pd.DataFrame({
    "Gerçek": y.tail(30).values,
    "Tahmin": y_pred[-30:]
})

st.line_chart(karsilastirma)


