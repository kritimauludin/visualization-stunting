import pickle
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler

# Load the model
stunting_model = pickle.load(open('stunting_model.sav', 'rb'))

# Tambahkan CSS untuk menebalkan teks pada sidebar
st.markdown("""
    <style>
    .nav-link p {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Membuat sidebar dengan option_menu
with st.sidebar:
    selected = option_menu(
        menu_title=None,  # required
        options=["Beranda", "Informasi", "Visualisasi Data", "Prediksi Stunting", "Visualisasi Penyebaran"],  # required
        icons=["house", "info-circle", "bar-chart", "activity", "map"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

# Tampilan berdasarkan pilihan menu
if selected == "Beranda":
    st.title("Selamat Datang Di Aplikasi Prediksi Stunting.")
    st.write("Aplikasi Prediksi Stunting adalah aplikasi untuk memprediksi kemungkinan balita mengalami stunting berdasarkan pada data rekam medik di Kabupaten Bogor. **Pilih menu pada sidebar** untuk mencoba prediksi")

elif selected == "Informasi":
    st.title("Informasi Stunting")

    st.subheader('Pengertian')
    st.write("Stunting pada balita adalah kondisi di mana anak mengalami pertumbuhan yang terhambat sehingga tinggi badannya lebih pendek dibandingkan dengan anak-anak seusianya. Stunting disebabkan oleh kekurangan gizi kronis selama periode kritis pertumbuhan dan perkembangan, terutama dalam 5 tahun kehidupannya.")
    
    st.subheader('Faktor')
    st.write("1. Gizi Buruk: Kekurangan asupan nutrisi yang memadai selama kehamilan dan masa bayi.")
    st.write("2. Penyakit Infeksi Berulang: Penyakit seperti diare dan infeksi saluran pernapasan yang sering..")
    st.write("3. Kurangnya Sanitasi dan Kebersihan: Lingkungan yang tidak bersih dapat menyebabkan infeksi berulang.")
    st.write("4. Kurangnya Akses ke Layanan Kesehatan: Termasuk kurangnya imunisasi dan perawatan kesehatan dasar.")
    st.write("5. Faktor Sosial dan Ekonomi: Kemiskinan, kurangnya pendidikan ibu, dan kondisi sosial lainnya.")

elif selected == "Visualisasi Data":
    st.title("Visualisasi Data")

    st.subheader("Visualisasi Barchart")
    st.write("Visualisasi barchart (grafik batang) adalah representasi data dalam bentuk batang horizontal atau vertikal. Setiap batang mewakili kategori atau nilai tertentu dan panjangnya sesuai dengan nilai data. Barchart berguna untuk membandingkan berbagai kategori data atau menunjukkan perubahan data dari waktu ke waktu.")
    
    st.write("1. Barchart")
    st.image('image/barchart.png', width=700)
    st.write("Pada Barchart ini menampilkan data stunting berbentuk grafik. Dapat diketahui bahwa Balita dengan kasus pendek (stunting) lebih banyak ditemukan dibandikan kasus sangat pendek (several stunting)")

    st.subheader("Featured Correlations")
    st.write("Featured Correlations berfungsi untuk mengetahui apakah diantara dua variabel terdapat hubungan atau tidak, dan jika ada hubungan bagaimanakah arah hubungan dan seberapa besar hubungan tersebut.")
    st.image('image/featured correlations.png', width=700)
    st.write('Dapat diketahui dari gambar diatas, bahwasanya jika hasil korelasi mendekati nilai 1, maka korelasi tersebut dikatakan baik, namun jika mendekati nilai -1 maka korelasi dikatakan buruk. Dapat dilihat dari hasil visualisasi di atas, yang memiliki nilai korelasi baik terdapat pada variabel "Usia", “Berat” dan “Tinggi” yaitu memiliki nilai hasil korelasinya 0.84, 0.89 dan 0.94, adapun nilai korelasi yang buruk terdapat pada variabel “BB/U” dan "ZS BB/U", yaitu memiliki nilai -0.83, jika hasil korelasi yang berbentuk diagonal menyamping tersebut itu merupakan korelasi variabel dengan dirinya sendiri.')

    st.subheader("Categorial Plot")
    st.write('Categorical plots adalah visualisasi seaborn dengan melakukan agregasi nilai per kolom yang berbentuk kategori.')
    st.image('image/categorial plot.jpg', width=700)
    st.write('Dapat diketahui dari gambar diatas, plot dilakukan pada setiap variabel berdasarkan TB/U sehingga membentuk suatu pola persebaran data pada masing-masing variabel. Pesebaran data balita yang terdiagnosis stunting rata-rata memiliki nilai yang tinggi pada variabel-variabel tersebut. Hal tersebut selaras dengan gambar correlations features, dimana variabel tersebut memiliki nilai yang besar dalam mempengaruhi balita berpotensi stunting.')
elif selected == "Prediksi Stunting":
    st.title("Prediksi Stunting")

    # Split the page into two columns
    col1, col2 = st.columns(2)

    with col1:
        Usia_Saat_Ukur = st.number_input('Input Usia (Bulan)', min_value=0, value=0, step=1)

    with col2:
        Berat = st.number_input('Input Berat (kg)', min_value=0.0, value=0.0, step=0.1)

    with col1:
        Tinggi = st.number_input('Input Tinggi (cm)', min_value=0.0, value=0.0, step=0.1)

    with col2:
        ZS_TB_U = st.number_input('Input ZS TB/U', min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

    # Code for prediction
    stunting_diagnosis = ''

    # Create a button for prediction
    if st.button('Test Prediksi Stunting'):
        stunting_prediction = stunting_model.predict([[Usia_Saat_Ukur, Berat, Tinggi, ZS_TB_U]])

        if stunting_prediction[0] == 1:
            stunting_diagnosis = 'Sangat Pendek'
        else:
            stunting_diagnosis = 'Pendek'

        st.success(stunting_diagnosis)

elif selected == "Visualisasi Penyebaran":
    st.title("Visualisasi Penyebaran Stunting")

    # Fungsi untuk memuat data dari file Excel
    def load_data(file_path):
        data = pd.read_excel(file_path)
        return data

    # Fungsi untuk visualisasi spasial
    def spatial_visualization(data):
        # Pisahkan kolom Latitude dan Longitude
        data['lat'] = pd.to_numeric(data['Latitude'], errors='coerce')
        data['lon'] = pd.to_numeric(data['Longitude'], errors='coerce')

        # Hapus baris dengan nilai latitude atau longitude yang tidak valid
        data = data.dropna(subset=['lat', 'lon'])

        # Konversi kolom Tanggal Pengukuran menjadi format datetime
        data['date'] = pd.to_datetime(data['Tanggal Pengukuran'], errors='coerce')

        # Filter data untuk baris di mana 'TB/U' yang termasuk status 'Pendek' atau 'Sangat Pendek'
        filtered_data = data[data['TB/U'].isin(['Pendek', 'Sangat Pendek'])]

        # Hitung frekuensi kemunculan setiap kombinasi lat dan lon
        location_counts = filtered_data.groupby(['lat', 'lon']).size().reset_index(name='count')

        # Hitung jumlah status 'Pendek' dan 'Sangat Pendek' di setiap lokasi
        status_counts = filtered_data.groupby(['lat', 'lon', 'TB/U']).size().unstack(fill_value=0)

        # Ambil desa dan puskesmas yang unik sebagai fungsi untuk setiap lokasi (lat, lon)
        location_info = filtered_data.groupby(['lat', 'lon']).agg({
            'Desa/Kel': 'first',
            'Pukesmas': 'first'
        }).reset_index()

        # Buat peta dasar dengan folium
        m = folium.Map(location=[filtered_data['lat'].mean(), filtered_data['lon'].mean()], zoom_start=12)

        # Tambahkan lingkaran untuk status 'Pendek' (merah) dan 'Sangat Pendek' (biru)
        for (lat, lon), status in status_counts.iterrows():
            count_pendek = status.get('Pendek', 0)
            count_sangat_pendek = status.get('Sangat Pendek', 0)

            # Pastikan nilai yang digunakan adalah tipe int biasa (bukan int64)
            count_pendek = int(count_pendek)
            count_sangat_pendek = int(count_sangat_pendek)

            # Cari desa dan puskesmas untuk lokasi tersebut
            location_row = location_info[(location_info['lat'] == lat) & (location_info['lon'] == lon)]
            desa = location_row['Desa/Kel'].values[0]
            pukesmas = location_row['Pukesmas'].values[0]

            # Tambahkan lingkaran untuk 'Pendek' dengan warna merah
            if count_pendek > 0:
                folium.Circle(
                    location=(lat, lon),
                    radius=count_pendek * 10,  # Radius yang ditentukan oleh jumlah 'Pendek'
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6
                ).add_to(m)

            # Tambahkan lingkaran untuk 'Sangat Pendek' dengan warna biru
            if count_sangat_pendek > 0:
                folium.Circle(
                    location=(lat, lon),
                    radius=count_sangat_pendek * 10,  # Radius yang ditentukan oleh jumlah 'Sangat Pendek'
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6
                ).add_to(m)

            # Buat teks popup dengan lokasi, desa, puskesmas, dan jumlah pendek dang sangat pendek
            popup_text = f"""
            *Lokasi* :({lat}, {lon}),
            *Desa* : {desa},
            *Pukesmas* : {pukesmas},
            *Jumlah Pendek* : {count_pendek},
            *Jumlah Sangat Pendek* : {count_sangat_pendek},
            *Total* : {count_pendek + count_sangat_pendek}
            """

            # Tambahkan marker dengan informasi popup
            folium.Marker(
                location=(lat, lon),
                popup=folium.Popup(popup_text, parse_html=True),
                icon=folium.Icon(color='green')
            ).add_to(m)

        # Tampilkan peta dalam Streamlit
        st_folium(m, width=700, height=500)

    # Upload file untuk visualisasi penyebaran
    uploaded_file = st.file_uploader("Upload file data stunting", type=['xlsx'])

    if uploaded_file:
        data = load_data(uploaded_file)
        spatial_visualization(data)
    else:
        st.write("Silakan upload file data terlebih dahulu.")
