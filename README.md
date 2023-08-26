# Data Pre-processing and Feature Engineering
Tahapan selanjutnya setelah melakukan *Exploratory Data Analysis* di Stage 1 adalah melakukan *Data Pre-processing* dan *Feature Engineering*. Tahapan ini sangatlah penting dilakukan sebelum melakukan modeling pada *Machine Learning*. 

Setelah melakukan eksplorasi data pada tahap sebelumnya telah didapatkan beragam pengetahuan mengenai dataset yang sedang kita olah dan penanganan apa saja yang mungkin kita lakukan terhadap dataset tersebut agar dataset lebih siap digunakan dan juga meningkatkan performa model nantinya.

Pada tahap *pre-processing* dan *feature engineering* sendiri dilakukan untuk membersihkan data, pada dasarnya data-data sangat kotor ketika belum diolah yang berdampak pada ke-akurasian data sehingga tahapan ini dilakukan penanganan tersebut. Penanganan yang dimaksud dapat dilakukan dengan,
Mengubah tipe/format data

* Membersihkan/imputasi data-data yang kosong.
* Menghilangkan data duplikasi yang tidak diinginkan.
* Menseleksi data/fitur yang redundant.
* Mengubah skala/distribusi data untuk mempermudah *learning*.
* Menambahkan data sintetis/duplikasi.
* Menambah *features* baru ataupun mengambil inti dari *feature*.
* Mereduksi *features* ke dalam dimensi yang lebih rendah.

Pada dasarnya proses ini berlangsung secara iteratif,

<p style="text-align: center;">Features Engineering -> Pre-Processing -> Machine Learning -> Model Evaluation</p>
<p style="text-align: center;"><--------------------------------------------------------------------</p>

<br>

# Dataset Description
Sebelum melanjutkan ke inti pembahasan, berikut disertakan kembali deskripsi mengenai dataset,

|**Kolom**|**Deskripsi**|
|:-------:|:-----------:|
|id|ID unik untuk setiap customer.|
|Gender|Jenis kelamin dari customer.
|Age|Usia customer.
|Driving_License|0 : customer tidak memiliki izin mengemudi, 1 : customer memiliki izin mengemudi.
|Region_Code|Kode unik untuk tiap wilayah customer. 
|Previously_Insured | 0 : Customer belum memiliki 'Asuransi Kendaraan', 1 : Customer sudah memiliki 'Asuransi Kendaraan'.
|Vehicle_Age | Usia dari kendaraan milik customer.
|Vehicle_Damage | 0 : Kendaraan customer belum pernah rusak, 1: Kendaraan customer sudah pernah rusak.
|Annual_Premium | Premi tahunan yang harus dibayar oleh customer.
|Policy_Sales_Channel | Kode channel/media yang digunakan untuk menghubungi customer.
|Vintage | Jumlah hari customer sudah bergabung dengan perusahaan.
|Response|0 : Customer tidak tertarik 'Asuransi Kendaraan', 0 : Customer tertarik 'Asuransi Kendaraan'.

<br>

# Prerequisites
1. Dataset download [`here`](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction?select=train.csv).
2. `pip install requirement.txt`.

<br>

# Table of Contents
- Data Preprocessing - Missing Values
- Data Preprocessing - Duplicated Values
- Data Preprocessing - Outliers
- Data Preprocessing - Feature Encoding
- Data Preprocessing - Class Imbalance
- Feature Engineering - Feature Selection
- Feature Engineering - Feature Extraction
- Feature Engineering - Feature Recommendation
- Data Preprocessing - Feature Transformation

## Data Preprocessing - Missing Values
Melakukan pengecekan missing values dengan menggunakan `.isna()`/`.isnull()` . Serta melakukan pengecekan karakter tertentu yang mungkin termasuk NaN/NULL. 

Setelah dilakukan pengecekan terhadap NULL/NaN values, tidak terdapat NULL/NaN pada dataset.

## Data Preprocessing - Duplicated Values

Melakukan pengecekan duplicated values dengan menggunakan `.duplicated()` dan juga mengecek susbet kolom id.

Setelah dilakukan pengecekan terhadap duplicated values, tidak terdapat data duplikasi pada dataset.

## Data Preprocessing - Outliers
Dari insight yang didapat ketika EDA yang telah dilakukan di Stage 1, kita tahu `Annual Premium` memiliki outliers yang cukup extreme sehingga ditangani dengan penghapusan IQR ataupun capping.

Setelah melakukan handling outliers dengan metode IQR dan Capping dan didapatkan hasil,
<p align="center">
    <img src="image/1.png", alt="outliersplot">
</p>

Diputuskan untuk tetap menggunakan dataframe df karena, kolom `Annual_Premium` merupakan hal yang normal jika terdapat outliers sehingga tidak dilakukan penghapusan outliers. Hal ini juga didasarkan dengan pertimbangan pembuatan model yang robust terhadap outliers.

## Data Preprocessing - Feature Encoding
Mengubah `Vehicle_Damage` ke integer dalam = 0: Kendaraan customer belum pernah rusak, 1: Kendaraan customer sudah pernah rusak, serta `Vehicle_Age` dam 0: < 1 Year, 1: 1-2 Years, 2: > 2 Years. Serta `Gender` dengan *One Hot Encoding*. Melakukan konversi ke angka mulai dari 0 untuk memudahkan kerja machine learning. Mengubah kolom dengan datatype bool ke integer agar lebih mudah diproses oleh model.

## Data Preprocessing - Class Imbalance
Penanganan Class Imbalance dilakukan dengan ***undersampling*** dengan pertimbangan agar data tidak cenderung bias, dimana selisih antara kedua value 0 dan 1 lebih dari 50% sehingga jika dilakukan *oversampling* tidak menjamin akan adanya peningkatan performansi machine learning.

## Feature Engineering - Feature Selection
Dilakukan penghapusan feature `id` yang tidak relevan terhadap model dengan `.drop()`. Lalu dilakukan pengecekan korelasi antar kolom dengan menggunakan heatmap.

<p align="center">
    <img src="image/2.png", alt="heatmap1">
</p>

Dari heatmap dapat didapatkan insight bahwa, `Age` dan `Vehicle_Age` merupakan kolom redundant sehingga diputuskan untuk tidak menggunakan pada kolom `Age` dengan pertimbangan kolom `Age` memiliki korelasi lebih kecil dibandingkan `Vehicle_Age`.

## Feature Engineering - Feature Extraction
*Feature Extraction* yang dibuat antara lain,
1. Age_Group, melakukan *dimension reduction* dengan mengelompokan feature `Age` menjadi 3 kategori utama dengan range YoungAdults 17 - 30 yang diwakili dengan 0, MiddleAged 31-45 diwakili dengan 1, OldAdults > 45 diwakili dengan 2.

2. Premium_cat, sama halnya dengan feature `Age_Group` feature ini adalah feature reduksi dari `Annual_Premium` dengan range LowPremium < 24406 diwakili dengan 0, MediumPremium 24406 - 61892.4 diwakili dengan 1 dan HighPremium > 61892.4 diwakili dengan 2.

3. Policy_cat, ekstraksi dari kolom Policy_Sales_Channel dengan menggunakan banyaknya customers dari tiap Channel yang kemudian diurutkan dari channel paling banyak dibagi menjadi 4 kategori.

4. Region_cat, sama halnya dengan Policy_cat feature ini merupakan ekstraksi dari kolom Region_Code dimana semakin banyak customers di Region tersebut maka akan masuk ke kategori yang lebih tinggi.

<p align="center">
    <img src="image/3.png", alt="heatmap2">
</p>

Dari hasil *feature extraction* didapatkan heatmap baru bahwa `Policy_cat` tidak memiliki korelasi dengan `Response`, sedangkan `Age_Group` dan `Premium_cat` memiliki korelasi *positive*.

## Feature Engineering - Feature Recommendation
1. `Premium_Per_Channel`, untuk menghitung dan memberi insight baru mengenai total premium dari berbagai `Policy_Sales_Channel` dengan begitu pengelompokkan Channel dapat dilakukan berdasarkan `Annual_Premium`.

2. `Vintage_Group`, feature baru yang mengubah feature `Vintage` menjadi kategori dengan range tertentu dimana diartikan menjadi New (baru bergabung), Intermediate (sudah bergabung cukup lama), Long-term (sudah bergabung lama).

3. `Not_Insured_and_Damaged`, kolom yang berisikan nilai 1 jika kolom `Previously_Insured` memiliki value 0 dan `Vehicle_Damage` memiliki value 1.

4. `Channel_Response_Rate`, merupakan rate respon dari tiap channel dimana menindikasikan seberapa efektif suatu channel untuk mendapatkan jawaban 'Yes' dari sini juga dapat dilakukan pengelompokkan Channels yang memiliki rate tinggi.

## Data Preprocessing - Feature Transformation
Melakukan transformasi terhadap kolom yang bukan merupakan kategori (numerik) namun merupakan kolom yang memang berupakan numerik (range). Sebelum dilakukan tranformasi dilakukan split test train dahulu untuk mencegah *Data Leakege*. Transformasi ini dilakukan dengan menggunakan metode boxcox.

Data Train
<p align="center">
    <img src="image/4.png", alt="datatrain">
</p>

Data Test
<p align="center">
    <img src="image/4.png", alt="datatest">
</p>

# Conclusion
Features yang dipilih `Region_Code`, `Vehicle_Age`, `Vehicle_Damage`, `Annual_Premium`, `Policy_Sales_Channel`, `Gen_Female`, `Gen_Male`, `Age_Group`, `Premium_cat`.

Sedangkan targetnya adalah `Response`.