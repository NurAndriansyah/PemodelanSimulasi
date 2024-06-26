import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator
from datetime import date
import matplotlib.pyplot as plt
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Judul Halaman
st.title('Prediksi Harga Saham')
st.subheader('Emiten : BMRI')
st.subheader('Kelompok 9:')
st.text('1. Nur Andriansyah (1301213475)')
st.text('2. Rey Reno Alvaro Ikhsan (1301213268)')
st.text('3. Rivan Fauzan (1301210554)')
st.text('4. Muhammad Fadli Ramadhan (1301203533)')

# Parameter Input
st.sidebar.header('Parameter Simulasi')
ticker = st.sidebar.text_input('Masukkan Kode Saham', 'BMRI.JK')
start_date = st.sidebar.date_input('Tanggal Mulai', date(2022, 6, 1))
end_date = st.sidebar.date_input('Tanggal Akhir', date(2024, 6, 1))
num_simulations = st.sidebar.number_input('Jumlah Simulasi', min_value=1, value=100)
time_horizon = st.sidebar.number_input('Jangka Waktu (Hari)', min_value=1, value=30)
seed = st.sidebar.number_input('Seed (opsional)', value=42)

# Pengumpulan data
try:
    logger.info("Starting data download...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    logger.info("Data download complete.")
except Exception as e:
    logger.error(f"An error occurred: {e}")
    st.error(f"An error occurred while downloading data: {e}")

# Proses data hanya jika berhasil diunduh
if not data.empty:
    data = data[['Close', 'Volume']].copy() 
    data.reset_index(inplace=True)

    # Cek missing values
    data.interpolate(method='linear', inplace=True)
    data.dropna(inplace=True)

    # Hitung return logaritmik harian
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # Hitung indikator teknikal
    bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['SMA'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
    data.dropna(inplace=True)

    # Fungsi untuk menjalankan simulasi GBM
    def run_gbm_simulation(data, num_simulations, time_horizon, seed):
        np.random.seed(seed)  # Set seed here for reproducibility
        last_price = data['Close'].iloc[-1]
        simulations = np.zeros((time_horizon, num_simulations))
        for i in range(num_simulations):
            prices = [last_price]
            for t in range(1, time_horizon):
                drift = np.mean(data['LogReturn'])
                volatility = np.std(data['LogReturn'])
                shock = np.random.normal(0, 1)
                price = prices[-1] * np.exp(drift + volatility * shock)
                prices.append(price)
            simulations[:, i] = prices
        return simulations

    # Jalankan simulasi
    simulations = run_gbm_simulation(data, num_simulations, time_horizon, seed)

    # Plot hasil simulasi
    st.subheader('Hasil Simulasi')
    fig, ax = plt.subplots()
    time_range = range(1, time_horizon + 1)
    for i in range(num_simulations):
        ax.plot(time_range, simulations[:, i], alpha=0.1, color='blue')
    ax.set_title('Simulasi Geometric Brownian Motion')
    ax.set_xlabel('Hari')
    ax.set_ylabel('Harga')
    st.pyplot(fig)

    # Tampilkan tabel data
    st.subheader('Data Historis')
    st.write(data.tail())

    # Simpan hasil simulasi ke file
    def save_simulation_to_file(simulations):
        df = pd.DataFrame(simulations)
        csv = df.to_csv(index=False)
        st.download_button(label="Download Hasil Simulasi (CSV)", data=csv, file_name='simulasi_gbm.csv', mime='text/csv')

    save_simulation_to_file(simulations)
else:
    st.error("Tidak ada data yang berhasil diunduh.")
