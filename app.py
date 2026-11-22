import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import math

st.set_page_config(page_title="Sistem Pemantau Stok UMKM", layout="wide")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ganti nama file sesuai file .pkl / .joblib Anda
    return joblib.load("model_prediksi_bakery.pkl")

model = load_model()

# --- 2. SIMULASI / LOAD DATABASE ---
# Dalam produksi, hubungkan ke SQLite atau Google Sheets
if "sales_df" not in st.session_state:
    st.session_state.sales_df = pd.DataFrame({
        "tanggal": pd.date_range(end=datetime.date.today(), periods=7).tolist() * 2,
        "Product": ["Bread"] * 7 + ["Chocolate Cake"] * 7,
        "terjual": [20, 22, 19, 25, 30, 28, 24, 10, 12, 11, 15, 18, 20, 16]
    })

if "stok_bahan" not in st.session_state:
    st.session_state.stok_bahan = pd.DataFrame({
        "id_bahan": ["BHN-01", "BHN-02", "BHN-03"],
        "nama_bahan": ["Tepung Terigu", "Gula Pasir", "Cokelat Bubuk"],
        "stok_saat_ini": [5000, 3000, 800],  # dalam gram
        "satuan": ["gram", "gram", "gram"],
        "kemasan_beli": [1000, 1000, 500]    # beli per 1kg / 500g
    })

if "resep_df" not in st.session_state:
    st.session_state.resep_df = pd.DataFrame({
        "Product": ["Bread", "Bread", "Chocolate Cake", "Chocolate Cake", "Chocolate Cake"],
        "id_bahan": ["BHN-01", "BHN-02", "BHN-01", "BHN-02", "BHN-03"],
        "takaran": [150, 20, 200, 100, 50]  # kebutuhan per 1 unit produk
    })

# --- 3. FUNGSI EKSTRAKSI FITUR & PREDIKSI ---
def generate_predictions(sales_df, target_date):
    produk_list = sales_df["Product"].unique()
    fitur_rows = []

    for prod in produk_list:
        df_p = sales_df[sales_df["Product"] == prod].sort_values("tanggal")
        
        # Ambil data historis untuk fitur lag
        penjualan_h1 = df_p["terjual"].iloc[-1] if len(df_p) >= 1 else 0
        penjualan_h7 = df_p["terjual"].iloc[-7] if len(df_p) >= 7 else penjualan_h1
        rata_h7 = df_p["terjual"].tail(7).mean() if len(df_p) > 0 else 0

        fitur_rows.append({
            "Product": prod,
            "hari_dalam_minggu": target_date.weekday(),
            "bulan": target_date.month,
            "penjualan_ke_1": penjualan_h1,
            "penjualan_ke_7": penjualan_h7,
            "rata_rata_h7": rata_h7
        })

    X_pred = pd.DataFrame(fitur_rows)
    # Prediksi menggunakan pipeline model Anda
    X_pred["prediksi_terjual"] = np.ceil(model.predict(X_pred))
    return X_pred

# --- 4. NAVIGASI ROLE (ADMIN vs OWNER) ---
st.sidebar.title("Navigasi Sistem")
role = st.sidebar.radio("Pilih Akses Pengguna:", ["Admin (Operasional)", "Owner (Pemilik)"])

# ==========================================
# TAMPILAN ADMIN
# ==========================================
if role == "Admin (Operasional)":
    st.title("Panel Operasional Admin")
    
    tab1, tab2 = st.tabs(["Input Penjualan Harian", "Update Stok Bahan Baku"])
    
    with tab1:
        st.subheader("Catat Penjualan Hari Ini")
        with st.form("form_penjualan"):
            tgl = st.date_input("Tanggal", datetime.date.today())
            produk = st.selectbox("Pilih Produk", [
                "Biscuit", "Black Forest Cake", "Bread", "Brown Bread", "Butter",
                "Chocolate Cake", "Coffee Powder", "Cookies", "Cupcake", "Donut",
                "Milk", "Muffin", "Pastry", "Puff", "Rusk", "Tea Cake"
            ])
            qty = st.number_input("Jumlah Terjual (Unit)", min_value=0, step=1)
            submit_jual = st.form_submit_button("Simpan Penjualan")
            
            if submit_jual:
                new_row = pd.DataFrame({"tanggal": [pd.to_datetime(tgl)], "Product": [produk], "terjual": [qty]})
                st.session_state.sales_df = pd.concat([st.session_state.sales_df, new_row], ignore_index=True)
                st.success(f"Data penjualan {produk} ({qty} unit) berhasil disimpan!")

        st.dataframe(st.session_state.sales_df.tail(10), use_container_width=True)

    with tab2:
        st.subheader("Stok Fisik Gudang Saat Ini")
        edited_stok = st.data_editor(st.session_state.stok_bahan, num_rows="dynamic", use_container_width=True)
        if st.button("Simpan Perubahan Stok"):
            st.session_state.stok_bahan = edited_stok
            st.success("Stok gudang diperbarui!")

# ==========================================
# TAMPILAN OWNER
# ==========================================
else:
    st.title("Dashboard Monitoring & Prediksi Owner")
    besok = datetime.date.today() + datetime.timedelta(days=1)
    st.write(f"Proyeksi kebutuhan untuk tanggal: **{besok.strftime('%d %B %Y')}**")

    # 1. Jalankan Prediksi Model
    df_hasil_pred = generate_predictions(st.session_state.sales_df, besok)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("1. Prediksi Penjualan Produk (ML)")
        st.dataframe(
            df_hasil_pred[["Product", "penjualan_ke_1", "rata_rata_h7", "prediksi_terjual"]],
            use_container_width=True
        )

    # 2. Konversi Prediksi Produk ke Kebutuhan Bahan Baku (BOM)
    kebutuhan_df = df_hasil_pred.merge(st.session_state.resep_df, on="Product")
    kebutuhan_df["total_butuh"] = kebutuhan_df["prediksi_terjual"] * kebutuhan_df["takaran"]
    rekap_bahan = kebutuhan_df.groupby("id_bahan")["total_butuh"].sum().reset_index()

    # 3. Bandingkan dengan Stok Gudang & Hitung Rekomendasi Beli
    rekomendasi = st.session_state.stok_bahan.merge(rekap_bahan, on="id_bahan", how="left").fillna(0)
    rekomendasi["selisih_stok"] = rekomendasi["stok_saat_ini"] - rekomendasi["total_butuh"]
    
    def hitung_beli(row):
        if row["selisih_stok"] < 0:
            kekurangan = abs(row["selisih_stok"])
            return math.ceil(kekurangan / row["kemasan_beli"])
        return 0

    rekomendasi["rekomendasi_beli_pack"] = rekomendasi.apply(hitung_beli, axis=1)
    rekomendasi["status"] = np.where(rekomendasi["selisih_stok"] < 0, "PERLU RESTOK", "AMAN")

    with col2:
        st.subheader("2. Rekomendasi Pembelian Bahan Baku")
        st.dataframe(
            rekomendasi[["nama_bahan", "stok_saat_ini", "total_butuh", "rekomendasi_beli_pack", "status"]],
            use_container_width=True
        )