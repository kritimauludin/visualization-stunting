import pickle
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
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
        options=["Beranda", "Informasi", "Visualisasi Data", "Klasifikasi Stunting", "Visualisasi Penyebaran"],  # required
        icons=["house", "info-circle", "bar-chart", "activity", "map"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

# Tampilan berdasarkan pilihan menu
if selected == "Beranda":
    st.title("Selamat Datang Di Aplikasi Prediksi Stunting.")
    st.write("Aplikasi Prediksi Stunting adalah aplikasi untuk memprediksi kemungkinan balita mengalami stunting berdasarkan pada data rekam medik di Kota Bogor. **Pilih menu pada sidebar** untuk mencoba prediksi")

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


elif selected == "Klasifikasi Stunting":
    st.title("Klasifikasi Stunting (Pendek dan Sangat Pendek)")

# Function to calculate the Z-score for stunting prediction (TB/U formula)
    who_reference_data = {
    "Laki-laki": {
        0: {"M": 49.88, "S": 1.83},
        1: {"M": 54.72, "S": 1.94},
        2: {"M": 58.42, "S": 2.00},
        3: {"M": 61.42, "S": 2.04},
        4: {"M": 63.88, "S": 2.08},
        5: {"M": 65.90, "S": 2.11},
        6: {"M": 67.62, "S": 2.14},
        7: {"M": 69.16, "S": 2.17},
        8: {"M": 70.59, "S": 2.20},
        9: {"M": 71.96, "S": 2.24},
        10: {"M": 73.28, "S": 2.28},
        11: {"M": 74.53, "S": 2.32},
        12: {"M": 75.74, "S": 2.37},
        13: {"M": 76.91, "S": 2.42},
        14: {"M": 78.04, "S": 2.47},
        15: {"M": 79.14, "S": 2.53},
        16: {"M": 80.21, "S": 2.58},
        17: {"M": 81.24, "S": 2.64},
        18: {"M": 82.25, "S": 2.69},
        19: {"M": 83.24, "S": 2.75},
        20: {"M": 84.19, "S": 2.81},
        21: {"M": 85.13, "S": 2.87},
        22: {"M": 86.04, "S": 2.93},
        23: {"M": 86.94, "S": 2.99},
        24: {"M": 87.81, "S": 3.05},
        25: {"M": 87.97, "S": 3.11},
        26: {"M": 88.80, "S": 3.17},
        27: {"M": 89.61, "S": 3.23},
        28: {"M": 90.41, "S": 3.29},
        29: {"M": 91.18, "S": 3.35},
        30: {"M": 91.93, "S": 3.40},
        31: {"M": 92.66, "S": 3.45},
        32: {"M": 93.37, "S": 3.51},
        33: {"M": 94.07, "S": 3.56},
        34: {"M": 94.75, "S": 3.61},
        35: {"M": 95.42, "S": 3.66},
        36: {"M": 96.08, "S": 3.70},
        37: {"M": 96.73, "S": 3.75},
        38: {"M": 97.37, "S": 3.79},
        39: {"M": 98.00, "S": 3.84},
        40: {"M": 98.63, "S": 3.88},
        41: {"M": 99.24, "S": 3.92},
        42: {"M": 99.85, "S": 3.96},
        43: {"M": 100.44, "S": 4.00},
        44: {"M": 101.03, "S": 4.04},
        45: {"M": 101.61, "S": 4.08},
        46: {"M": 102.19, "S": 4.11},
        47: {"M": 102.76, "S": 4.15},
        48: {"M": 103.32, "S": 4.19},
        49: {"M": 103.88, "S": 4.23},
        50: {"M": 104.44, "S": 4.26},
        51: {"M": 105.00, "S": 4.30},
        52: {"M": 105.55, "S": 4.34},
        53: {"M": 106.11, "S": 4.37},
        54: {"M": 106.66, "S": 4.41},
        55: {"M": 107.21, "S": 4.45},
        56: {"M": 107.76, "S": 4.48},
        57: {"M": 108.31, "S": 4.52},
        58: {"M": 108.86, "S": 4.56},
        59: {"M": 109.41, "S": 4.59},
        60: {"M": 109.96, "S": 4.63},
    },
    "Perempuan": {
        0: {"M": 49.14, "S": 1.86},
        1: {"M": 53.68, "S": 1.95},
        2: {"M": 57.06, "S": 2.03},
        3: {"M": 59.80, "S": 2.10},
        4: {"M": 62.08, "S": 2.16},
        5: {"M": 64.03, "S": 2.21},
        6: {"M": 65.73, "S": 2.26},
        7: {"M": 67.28, "S": 2.31},
        8: {"M": 68.74, "S": 2.36},
        9: {"M": 70.14, "S": 2.41},
        10: {"M": 71.48, "S": 2.46},
        11: {"M": 72.77, "S": 2.52},
        12: {"M": 74.01, "S": 2.57},
        13: {"M": 75.21, "S": 2.62},
        14: {"M": 76.38, "S": 2.68},
        15: {"M": 77.50, "S": 2.73},
        16: {"M": 78.60, "S": 2.79},
        17: {"M": 79.67, "S": 2.84},
        18: {"M": 80.70, "S": 2.90},
        19: {"M": 81.71, "S": 2.95},
        20: {"M": 82.70, "S": 3.01},
        21: {"M": 83.66, "S": 3.06},
        22: {"M": 84.60, "S": 3.12},
        23: {"M": 85.52, "S": 3.17},
        24: {"M": 86.41, "S": 3.22},
        25: {"M": 86.59, "S": 3.27},
        26: {"M": 87.44, "S": 3.33},
        27: {"M": 88.28, "S": 3.38},
        28: {"M": 89.10, "S": 3.43},
        29: {"M": 89.89, "S": 3.48},
        30: {"M": 90.67, "S": 3.53},
        31: {"M": 91.44, "S": 3.57},
        32: {"M": 92.19, "S": 3.62},
        33: {"M": 92.92, "S": 3.67},
        34: {"M": 93.64, "S": 3.71},
        35: {"M": 94.35, "S": 3.76},
        36: {"M": 95.05, "S": 3.80},
        37: {"M": 95.73, "S": 3.85},
        38: {"M": 96.41, "S": 3.89},
        39: {"M": 97.08, "S": 3.93},
        40: {"M": 97.74, "S": 3.98},
        41: {"M": 98.40, "S": 4.02},
        42: {"M": 99.04, "S": 4.06},
        43: {"M": 99.67, "S": 4.10},
        44: {"M": 100.30, "S": 4.14},
        45: {"M": 100.92, "S": 4.18},
        46: {"M": 101.53, "S": 4.22},
        47: {"M": 102.13, "S": 4.26},
        48: {"M": 102.73, "S": 4.30},
        49: {"M": 103.31, "S": 4.34},
        50: {"M": 103.90, "S": 4.38},
        51: {"M": 104.47, "S": 4.42},
        52: {"M": 105.04, "S": 4.46},
        53: {"M": 105.61, "S": 4.49},
        54: {"M": 106.17, "S": 4.53},
        55: {"M": 106.72, "S": 4.57},
        56: {"M": 107.27, "S": 4.61},
        57: {"M": 107.82, "S": 4.64},
        58: {"M": 108.36, "S": 4.68},
        59: {"M": 108.89, "S": 4.71},
        60: {"M": 109.42, "S": 4.75},
    }
}

# Function to calculate the Z-score for stunting prediction (TB/U formula)
    def calculate_zscore(gender, age, height):
        if age not in who_reference_data[gender]:
            return None  # No data available for this age

    # Get the reference values for the selected age and gender
        M = who_reference_data[gender][age]["M"]
        S = who_reference_data[gender][age]["S"]
    
    # Calculate the Z-score
        z_score = (height - M) / S
        return z_score

# Define the stunting diagnosis based on the Z-score
    def diagnose_stunting(z_score):
        if z_score < -3:
            return 'Sangat Pendek'  # Severe Stunting (Z < -3)
        elif z_score < -2:
            return 'Pendek'  # Stunting (Z between -2 and -3)
        else:
            return 'Tidak Stunting'  # Normal Growth (Z >= -2)

# Split the page into two columns
    col1, col2 = st.columns(2)

# User inputs
    with col1:
        usia = st.number_input('Input Usia (Bulan)', min_value=0, max_value=60, value=0, step=1)

    with col2:
        gender = st.selectbox('Pilih Jenis Kelamin', ['Laki-laki', 'Perempuan'])

    with col1:
        tinggi = st.number_input('Input Tinggi (cm)', min_value=0.0, max_value=105.0, value=0.0, step=0.1)

# Code for prediction
    stunting_diagnosis = ''

# Create a button for prediction
    if st.button('Test Prediksi Stunting'):
    # Calculate the Z-score
        z_score = calculate_zscore(gender, usia, tinggi)
    
        if z_score is None:
            st.warning(f'Tidak ada data referensi untuk usia {usia} bulan pada jenis kelamin {gender}.')
        else:
        # Diagnose based on Z-score
            stunting_diagnosis = diagnose_stunting(z_score)

        # Display the Z-score and diagnosis
            st.write(f'Z-score (ZS TB/U): {z_score:.2f}')
            st.success(stunting_diagnosis)


elif selected == "Visualisasi Penyebaran":
    st.title("Visualisasi Penyebaran Stunting")

    # Fungsi untuk memuat data dari file Excel
    def load_data(file_path):
        data = pd.read_excel(file_path)
        return data

    # Fungsi untuk visualisasi spasial
    def spatial_visualization(data):
        data['lat'] = pd.to_numeric(data['Latitude'], errors='coerce')
        data['lon'] = pd.to_numeric(data['Longitude'], errors='coerce')
        data = data.dropna(subset=['lat', 'lon'])
        data['date'] = pd.to_datetime(data['Tanggal Pengukuran'], errors='coerce')
        
        filtered_data = data[data['TB/U'].isin(['Pendek', 'Sangat Pendek'])]
        location_counts = filtered_data.groupby(['lat', 'lon']).size().reset_index(name='count')
        status_counts = filtered_data.groupby(['lat', 'lon', 'TB/U']).size().unstack(fill_value=0)
        location_info = filtered_data.groupby(['lat', 'lon']).agg({
            'Desa/Kel': 'first',
            'Pukesmas': 'first'
        }).reset_index()
        
        m = folium.Map(location=[filtered_data['lat'].mean(), filtered_data['lon'].mean()], zoom_start=12)
        
        for (lat, lon), status in status_counts.iterrows():
            count_pendek = int(status.get('Pendek', 0))
            count_sangat_pendek = int(status.get('Sangat Pendek', 0))
            location_row = location_info[(location_info['lat'] == lat) & (location_info['lon'] == lon)]
            desa = location_row['Desa/Kel'].values[0]
            pukesmas = location_row['Pukesmas'].values[0]

            if count_pendek > 0:
                folium.Circle(
                    location=(lat, lon),
                    radius=count_pendek * 10,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6
                ).add_to(m)

            if count_sangat_pendek > 0:
                folium.Circle(
                    location=(lat, lon),
                    radius=count_sangat_pendek * 10,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6
                ).add_to(m)
            
            popup_text = f"""
            Lokasi: ({lat}, {lon}) | 
            Desa: {desa} | 
            Puskesmas: {pukesmas} | 
            Pendek: {count_pendek} | 
            Sangat Pendek: {count_sangat_pendek} | 
            Total: {count_pendek + count_sangat_pendek}
            """
            folium.Marker(
                location=(lat, lon),
                popup=folium.Popup(popup_text,  max_width=450, parse_html=True),
                icon=folium.Icon(color='green')
            ).add_to(m)
        
        st_folium(m, width=700, height=500)

        st.markdown("---")

    # Fungsi untuk menampilkan diagram batang berdasarkan Tahun
    def line_chart(data):
        if 'Tahun' in data.columns and 'Kec' in data.columns:
            # Hitung jumlah total kasus per tahun
            total_per_year = data.groupby('Tahun').size().reset_index(name='Total Kasus')
            # Hitung jumlah kasus per tahun dan kecamatan
            count_data = data.groupby(['Tahun', 'Kec']).size().reset_index(name='Jumlah Kasus')
            # Gabungkan dengan total per tahun untuk menghitung persentase
            count_data = count_data.merge(total_per_year, on='Tahun')
            count_data['Persentase Kasus'] = (count_data['Jumlah Kasus'] / count_data['Total Kasus']) * 100

            # Buat diagram garis dengan warna berdasarkan Kecamatan
            fig = px.line(
                count_data, x='Tahun', y='Persentase Kasus', color='Kec',
                title='Prevalensi Stunting di Kota Bogor per Kecamatan',
                labels={'Tahun': 'Tahun', 'Persentase Kasus': 'Persentase (%)', 'Kec': 'Kec'},
                markers=True
            )
            fig.update_xaxes(tickmode='linear', tickformat='d')
            fig.update_layout(yaxis_tickformat=".2f")  
            st.plotly_chart(fig)

            pivot_table = count_data.pivot(index='Tahun', columns='Kec', values='Persentase Kasus').fillna(0)
            pivot_table['Total Kasus'] = total_per_year.set_index('Tahun')['Total Kasus'].astype(int)  
            pivot_table.index = pivot_table.index.astype(int)
            pivot_table = pivot_table.T
            pivot_table_display = pivot_table.applymap(lambda x: f"{x:.2f}%" if isinstance(x, float) else f"{x:,}")
            pivot_table_display.loc['Total Kasus'] = pivot_table.loc['Total Kasus'].apply(lambda x: f"{int(x):,}")
            st.dataframe(pivot_table_display)

        else:
            st.write("Kolom 'Tahun' atau 'Kecamatan' tidak ditemukan dalam dataset.")

        st.markdown("---")

# Fungsi untuk menampilkan diagram batang dengan filter kategori
    def bar_chart(data):
        if 'Tahun' in data.columns and 'Desa/Kel' in data.columns and 'TB/U' in data.columns:
            # Pilihan tahun
            tahun_options = sorted(data['Tahun'].unique())
            selected_tahun = st.selectbox("Pilih Tahun:", tahun_options, index=len(tahun_options)-1)

            # Filter berdasarkan tahun
            data = data[data['Tahun'] == selected_tahun]

            # Pilihan kategori
            kategori_options = ['Pendek', 'Sangat Pendek']
            selected_kategori = st.multiselect("Pilih Kategori Stunting:", kategori_options, default=kategori_options)

            # Filter berdasarkan kategori
            filtered_data = data[data['TB/U'].isin(selected_kategori)]

            if filtered_data.empty:
                st.write("Tidak ada data yang tersedia untuk filter yang dipilih.")
                return

            # Hitung jumlah kasus per Desa/Kel
            count_data = filtered_data.groupby(['Desa/Kel', 'TB/U']).size().reset_index(name='Jumlah Kasus')

            # Warna kategori
            color_map = {'Pendek': 'red', 'Sangat Pendek': 'blue'}

            # Buat diagram batang
            fig = px.bar(
                count_data,
                x='Desa/Kel',  
                y='Jumlah Kasus',  
                color='TB/U',  
                title=f'Jumlah Kasus Stunting per Desa/Kelurahan (Tahun {selected_tahun})',
                labels={'Jumlah Kasus': 'Jumlah Kasus', 'Desa/Kel': 'Desa/Kelurahan', 'TB/U': 'Kategori'},
                color_discrete_map=color_map,
                text_auto=True  
            )

            # Atur tampilan agar lebih optimal
            fig.update_xaxes(
                categoryorder="total descending",
                tickangle=-45,  
                tickfont=dict(size=6),  
                automargin=True
            )
            fig.update_layout(width=2000, height=800)  

            st.plotly_chart(fig)

        else:
            st.write("Kolom 'Tahun', 'Desa/Kel', atau 'TB/U' tidak ditemukan dalam dataset.")

# Fungsi untuk memuat data dari file Excel
    def load_data(file_path):
        return pd.read_excel(file_path)

# Inisialisasi session state jika belum ada
    if 'data' not in st.session_state:
        st.session_state['data'] = None

    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

# Upload file untuk visualisasi penyebaran
    uploaded_file = st.file_uploader("Upload file data stunting", type=['xlsx'])

# Jika pengguna mengunggah file, simpan ke session_state dan load datanya
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['data'] = load_data(uploaded_file)

# Gunakan data dari session_state jika sudah ada
    data = st.session_state['data']

# Pastikan data tidak None sebelum digunakan
    if data is not None:
        st.write("✅ File telah diunggah dan diproses!")
        spatial_visualization(data)
        line_chart(data)
        bar_chart(data)
    else:
        st.write("❌ Silakan upload file data terlebih dahulu.")
