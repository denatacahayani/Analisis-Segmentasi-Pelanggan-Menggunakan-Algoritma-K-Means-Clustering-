import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Analisis Segmentasi Pelanggan",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("📊 Menu")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Beranda",
        "📁 Dataset",
        "🧹 Data Cleaning",
        "📈 Visualisasi Data",
        "📉 Elbow Method",
        "🎯 Clustering",
        "📊 Hasil Cluster",
        "📝 Kesimpulan"
    ]
)

# ==================================================
# BERANDA
# ==================================================
if menu == "🏠 Beranda":

    st.title("📊 Analisis Segmentasi Pelanggan")

    st.subheader(
        "Menggunakan Algoritma K-Means Clustering Berdasarkan Pendapatan Tahunan dan Spending Score"
    )

    st.markdown("---")

    st.write("""
    ### Deskripsi

    Aplikasi ini digunakan untuk melakukan segmentasi pelanggan menggunakan algoritma K-Means Clustering.

    Variabel yang digunakan:

    - Annual Income (Pendapatan Tahunan)
    - Spending Score (Skor Pengeluaran)

    Dataset yang digunakan adalah Mall Customers Dataset.

    ### Tujuan

    Mengelompokkan pelanggan berdasarkan karakteristik pendapatan dan pola pengeluaran sehingga dapat membantu pengambilan keputusan pemasaran.
    """)

# ==================================================
# MENU SELAIN BERANDA MEMBUTUHKAN DATASET
# ==================================================
else:

    if uploaded_file is None:
        st.warning("Silakan upload dataset CSV terlebih dahulu melalui sidebar.")
        st.stop()

    df = pd.read_csv(uploaded_file)

    # ==================================================
    # DATASET
    # ==================================================
    if menu == "📁 Dataset":

        st.title("📁 Dataset")

        st.dataframe(df)

        st.subheader("Informasi Dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Jumlah Baris", df.shape[0])

        with col2:
            st.metric("Jumlah Kolom", df.shape[1])

        with col3:
            st.metric("Missing Value", df.isnull().sum().sum())

    # ==================================================
    # DATA CLEANING
    # ==================================================
    elif menu == "🧹 Data Cleaning":

        st.title("🧹 Data Cleaning")

        st.subheader("Missing Value")

        st.dataframe(df.isnull().sum())

        st.subheader("Data Duplikat")

        duplicate = df.duplicated().sum()

        st.write(f"Jumlah data duplikat: {duplicate}")

        st.subheader("Tipe Data")

        dtype_df = pd.DataFrame(df.dtypes, columns=["Tipe Data"])

        st.dataframe(dtype_df)

    # ==================================================
    # VISUALISASI DATA
    # ==================================================
    elif menu == "📈 Visualisasi Data":

        st.title("📈 Visualisasi Data")

        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            X['Annual Income (k$)'],
            X['Spending Score (1-100)']
        )

        ax.set_title("Sebaran Data Pelanggan")
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")

        st.pyplot(fig)

    # ==================================================
    # ELBOW METHOD
    # ==================================================
    elif menu == "📉 Elbow Method":

        st.title("📉 Elbow Method")

        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

        wcss = []

        for i in range(1, 11):

            model = KMeans(
                n_clusters=i,
                random_state=42,
                n_init=10
            )

            model.fit(X)

            wcss.append(model.inertia_)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            range(1, 11),
            wcss,
            marker='o'
        )

        ax.set_title("Grafik Elbow Method")
        ax.set_xlabel("Jumlah Cluster")
        ax.set_ylabel("WCSS")

        st.pyplot(fig)

        st.info(
            "Berdasarkan grafik Elbow Method, jumlah cluster optimal biasanya berada pada k = 5."
        )

    # ==================================================
    # CLUSTERING
    # ==================================================
    elif menu == "🎯 Clustering":

        st.title("🎯 K-Means Clustering")

        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

        k = st.slider(
            "Pilih Jumlah Cluster",
            min_value=2,
            max_value=10,
            value=5
        )

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        cluster = model.fit_predict(X)

        df["Cluster"] = cluster

        st.success("Proses clustering berhasil.")

        st.subheader("Data Hasil Clustering")

        st.dataframe(df)

        st.subheader("Titik Centroid")

        centroid = pd.DataFrame(
            model.cluster_centers_,
            columns=[
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        )

        st.dataframe(centroid)

    # ==================================================
    # HASIL CLUSTER
    # ==================================================
    elif menu == "📊 Hasil Cluster":

        st.title("📊 Hasil Visualisasi Cluster")

        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

        model = KMeans(
            n_clusters=5,
            random_state=42,
            n_init=10
        )

        cluster = model.fit_predict(X)

        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(
            X['Annual Income (k$)'],
            X['Spending Score (1-100)'],
            c=cluster
        )

        ax.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            s=300,
            marker='X'
        )

        ax.set_title("Visualisasi Hasil K-Means Clustering")
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")

        st.pyplot(fig)

        cluster_count = pd.Series(cluster).value_counts().sort_index()

        st.subheader("Jumlah Anggota Setiap Cluster")

        st.dataframe(cluster_count)

    # ==================================================
    # KESIMPULAN
    # ==================================================
    elif menu == "📝 Kesimpulan":

        st.title("📝 Kesimpulan")

        st.write("""
        Berdasarkan hasil clustering menggunakan algoritma K-Means,
        pelanggan dapat dikelompokkan ke dalam beberapa segmen berdasarkan
        pendapatan tahunan dan spending score.

        Segmentasi pelanggan dapat membantu perusahaan dalam:

        - Menentukan target pasar.
        - Menyusun strategi promosi.
        - Meningkatkan loyalitas pelanggan.
        - Mengoptimalkan layanan berdasarkan karakteristik pelanggan.

        Algoritma K-Means berhasil mengelompokkan pelanggan dengan karakteristik yang serupa ke dalam cluster yang sama.
        """)