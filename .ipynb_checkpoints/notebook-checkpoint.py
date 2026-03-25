# ============================================================
# Proyek Analisis Data: E-Commerce Public Dataset
# - Nama       : Hayqal Akbar Rizky Iskandar
# - Email      : cdcc179d6y0826@student.devacademy.id
# - ID Dicoding: Hayqal Akbar Rizky Iskandar
# ============================================================

# ============================================================
# Menentukan Pertanyaan Bisnis
# ============================================================
# - Pertanyaan 1: Bagaimana tren jumlah pesanan (order) bulanan dari tahun 2016 hingga 2018,
#                 dan bulan apa yang memiliki jumlah pesanan tertinggi?
# - Pertanyaan 2: Kategori produk apa yang paling banyak terjual dan menghasilkan pendapatan tertinggi?
# - Pertanyaan 3: Bagaimana distribusi geografis pelanggan berdasarkan state di Brasil?
# - Pertanyaan 4: Siapa saja pelanggan terbaik berdasarkan analisis RFM (Recency, Frequency, Monetary)?

# ============================================================
# Import Semua Packages/Library yang Digunakan
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============================================================
# Load Dataset
# ============================================================

DATA_PATH = "data/"

customers_df      = pd.read_csv(DATA_PATH + 'customers_dataset.csv')
geolocation_df    = pd.read_csv(DATA_PATH + 'geolocation_dataset.csv')
order_items_df    = pd.read_csv(DATA_PATH + 'order_items_dataset.csv')
order_payments_df = pd.read_csv(DATA_PATH + 'order_payments_dataset.csv')
order_reviews_df  = pd.read_csv(DATA_PATH + 'order_reviews_dataset.csv')
orders_df         = pd.read_csv(DATA_PATH + 'orders_dataset.csv')
products_df       = pd.read_csv(DATA_PATH + 'products_dataset.csv')
sellers_df        = pd.read_csv(DATA_PATH + 'sellers_dataset.csv')
category_df       = pd.read_csv(DATA_PATH + 'product_category_name_translation.csv')

print(f'customers_df      : {customers_df.shape}')
print(f'geolocation_df    : {geolocation_df.shape}')
print(f'order_items_df    : {order_items_df.shape}')
print(f'order_payments_df : {order_payments_df.shape}')
print(f'order_reviews_df  : {order_reviews_df.shape}')
print(f'orders_df         : {orders_df.shape}')
print(f'products_df       : {products_df.shape}')
print(f'sellers_df        : {sellers_df.shape}')
print(f'category_df       : {category_df.shape}')

# ============================================================
# Data Wrangling
# ============================================================

# ----------------------------
# Gathering Data
# ----------------------------

# Preview masing-masing dataset
print(' customers_df ')
display(customers_df.head(3))

print(' orders_df ')
display(orders_df.head(3))

print(' order_items_df ')
display(order_items_df.head(3))

print(' products_df ')
display(products_df.head(3))

print(' geolocation_df ')
display(geolocation_df.head(3))

# Insight:
# - Dataset E-Commerce terdiri dari 9 tabel yang saling terhubung melalui foreign key
#   seperti order_id, customer_id, product_id, dan seller_id.
# - Dataset mencakup informasi transaksi, produk, pelanggan, penjual, dan geolokasi di Brasil.

# ----------------------------
# Assessing Data
# ----------------------------

# Memeriksa missing values dan tipe data pada setiap dataset
datasets = {
    'customers_df'      : customers_df,
    'geolocation_df'    : geolocation_df,
    'order_items_df'    : order_items_df,
    'order_payments_df' : order_payments_df,
    'order_reviews_df'  : order_reviews_df,
    'orders_df'         : orders_df,
    'products_df'       : products_df,
    'sellers_df'        : sellers_df,
    'category_df'       : category_df
}

for name, df in datasets.items():
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    duplicates = df.duplicated().sum()
    print(f'\n=== {name} ===')
    print(f'Shape        : {df.shape}')
    print(f'Duplicates   : {duplicates}')
    if len(missing) > 0:
        print(f'Missing Values:')
        print(missing)
    else:
        print('Missing Values: None')

# Memeriksa tipe data orders_df (kolom datetime perlu dikonversi)
print(orders_df.dtypes)
print()
print(orders_df.describe())

# Insight:
# - Terdapat missing values pada beberapa kolom di orders_df (kolom timestamp pengiriman)
#   dan products_df (nama kategori).
# - Kolom-kolom bertipe datetime masih berformat string sehingga perlu dikonversi.
# - Terdapat duplikat pada geolocation_df yang perlu dihapus.

# ----------------------------
# Cleaning Data
# ----------------------------

# 1. Konversi kolom datetime pada orders_df
datetime_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
for col in datetime_cols:
    orders_df[col] = pd.to_datetime(orders_df[col])

print('Tipe data setelah konversi:')
print(orders_df[datetime_cols].dtypes)

# 2. Hapus duplikat pada geolocation_df
print(f'Sebelum drop duplikat: {geolocation_df.shape}')
geolocation_df = geolocation_df.drop_duplicates()
print(f'Setelah drop duplikat : {geolocation_df.shape}')

# 3. Handle missing values pada products_df — isi kategori yang kosong dengan 'unknown'
print(f'Missing sebelum: {products_df["product_category_name"].isnull().sum()}')
products_df['product_category_name'] = products_df['product_category_name'].fillna('unknown')
print(f'Missing setelah : {products_df["product_category_name"].isnull().sum()}')

# 4. Filter orders hanya yang berstatus 'delivered'
print(f'Status order unik: {orders_df["order_status"].unique()}')
orders_delivered = orders_df[orders_df['order_status'] == 'delivered'].copy()
print(f'Jumlah order delivered: {orders_delivered.shape[0]}')

# 5. Gabungkan products_df dengan terjemahan kategori
products_df = products_df.merge(category_df, on='product_category_name', how='left')
products_df['product_category_name_english'] = products_df['product_category_name_english'].fillna('unknown')
print('Produk dengan kategori Bahasa Inggris:')
display(products_df[['product_id', 'product_category_name', 'product_category_name_english']].head())

# Insight:
# - Kolom datetime pada orders_df berhasil dikonversi ke tipe datetime.
# - Duplikat pada geolocation_df berhasil dihapus.
# - Missing values pada kolom product_category_name diisi dengan label 'unknown'.
# - Hanya order dengan status 'delivered' yang digunakan untuk analisis agar hasil lebih akurat.

# ============================================================
# Export Clean Dataset for Dashboard
# ============================================================

main_data = orders_delivered.merge(customers_df, on='customer_id', how='left')

main_data = main_data.merge(
    order_items_df[['order_id', 'product_id', 'price']],
    on='order_id',
    how='left'
)

main_data = main_data.merge(
    products_df[['product_id', 'product_category_name_english']],
    on='product_id',
    how='left'
)

main_data = main_data.merge(
    order_payments_df.groupby('order_id')['payment_value'].sum().reset_index(),
    on='order_id',
    how='left'
)

main_data.to_csv('dashboard/main_data.csv', index=False)

print('main_data.csv berhasil dibuat')
print(main_data.shape)

# ============================================================
# Exploratory Data Analysis (EDA)
# ============================================================

# ----------------------------
# Explore Tren Pesanan Bulanan
# ----------------------------

# Tambahkan kolom year_month untuk analisis tren
orders_delivered['year_month'] = orders_delivered['order_purchase_timestamp'].dt.to_period('M')

monthly_orders = orders_delivered.groupby('year_month').agg(
    total_orders=('order_id', 'count')
).reset_index()

monthly_orders['year_month_str'] = monthly_orders['year_month'].astype(str)

print('Tren Pesanan Bulanan:')
display(monthly_orders.sort_values('total_orders', ascending=False).head(10))

# Insight:
# - Terjadi peningkatan jumlah pesanan yang signifikan dari tahun 2016 ke 2018.
# - Bulan-bulan tertentu menunjukkan lonjakan pesanan yang kemungkinan berkaitan
#   dengan event promosi atau musim belanja.

# ----------------------------
# Explore Kategori Produk Terlaris
# ----------------------------

# Gabungkan order_items dengan products dan orders
items_products = order_items_df.merge(
    products_df[['product_id', 'product_category_name_english']],
    on='product_id', how='left'
)

items_products = items_products.merge(
    orders_delivered[['order_id']],
    on='order_id', how='inner'
)

# Hitung penjualan dan revenue per kategori
category_sales = items_products.groupby('product_category_name_english').agg(
    total_sold=('order_id', 'count'),
    total_revenue=('price', 'sum')
).reset_index().sort_values('total_sold', ascending=False)

print('Top 10 Kategori Produk Terlaris:')
display(category_sales.head(10))

# Insight:
# - Kategori bed_bath_table, health_beauty, dan sports_leisure mendominasi penjualan.
# - Kategori dengan penjualan tertinggi belum tentu menghasilkan revenue tertinggi
#   karena perbedaan harga produk.

# ----------------------------
# Explore Distribusi Geografis Pelanggan
# ----------------------------

# Distribusi pelanggan berdasarkan state
customer_state = customers_df.groupby('customer_state').agg(
    total_customers=('customer_id', 'count')
).reset_index().sort_values('total_customers', ascending=False)

print('Distribusi Pelanggan per State (Top 10):')
display(customer_state.head(10))

# Insight:
# - Pelanggan paling banyak berasal dari state SP (Sao Paulo), diikuti RJ (Rio de Janeiro)
#   dan MG (Minas Gerais).
# - Konsentrasi pelanggan di wilayah tenggara Brasil mencerminkan pusat ekonomi negara tersebut.

# ----------------------------
# Explore RFM Analysis
# ----------------------------

# Gabungkan orders dengan customers dan payments
orders_customers = orders_delivered.merge(customers_df, on='customer_id', how='left')
orders_payments  = orders_customers.merge(
    order_payments_df.groupby('order_id')['payment_value'].sum().reset_index(),
    on='order_id', how='left'
)

# Tentukan tanggal referensi
reference_date = orders_payments['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

rfm_df = orders_payments.groupby('customer_unique_id').agg(
    Recency=('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
    Frequency=('order_id', 'count'),
    Monetary=('payment_value', 'sum')
).reset_index()

print('RFM DataFrame:')
display(rfm_df.describe())

# Scoring RFM menggunakan binning (1–5)
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'],   q=5, labels=[5,4,3,2,1])
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'],  q=5, labels=[1,2,3,4,5])

rfm_df['RFM_Score'] = (
    rfm_df['R_Score'].astype(int) +
    rfm_df['F_Score'].astype(int) +
    rfm_df['M_Score'].astype(int)
)

# Segmentasi pelanggan
def segment_customer(score):
    if score >= 13:
        return 'Champions'
    elif score >= 10:
        return 'Loyal Customers'
    elif score >= 7:
        return 'Potential Loyalists'
    elif score >= 4:
        return 'At Risk'
    else:
        return 'Lost'

rfm_df['Segment'] = rfm_df['RFM_Score'].apply(segment_customer)

print('Distribusi Segmen Pelanggan:')
print(rfm_df['Segment'].value_counts())

# Insight:
# - Sebagian besar pelanggan termasuk dalam segmen 'At Risk' dan 'Potential Loyalists'.
# - Pelanggan kategori 'Champions' memiliki frekuensi pembelian tinggi dan pengeluaran terbesar.

# ============================================================
# Visualization & Explanatory Analysis
# ============================================================

# ----------------------------
# Pertanyaan 1: Bagaimana tren jumlah pesanan bulanan dari tahun 2016 hingga 2018?
# ----------------------------

fig, ax = plt.subplots(figsize=(16, 6))

x = range(len(monthly_orders))
ax.plot(x, monthly_orders['total_orders'], marker='o', color='#1a73e8', linewidth=2.5, markersize=6)
ax.fill_between(x, monthly_orders['total_orders'], alpha=0.15, color='#1a73e8')

ax.set_xticks(x)
ax.set_xticklabels(monthly_orders['year_month_str'], rotation=45, ha='right', fontsize=9)
ax.set_title('Tren Jumlah Pesanan Bulanan (2016–2018)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Bulan', fontsize=12)
ax.set_ylabel('Jumlah Pesanan', fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()

# Insight:
# - Jumlah pesanan mengalami pertumbuhan yang konsisten dari 2016 hingga akhir 2017,
#   kemudian mencapai puncaknya.
# - Terdapat lonjakan signifikan pada bulan November 2017 yang kemungkinan besar berkaitan
#   dengan event Black Friday / Harbolnas.

# ----------------------------
# Pertanyaan 2: Kategori produk apa yang paling banyak terjual dan menghasilkan pendapatan tertinggi?
# ----------------------------

top_n = 10
top_sold    = category_sales.head(top_n).sort_values('total_sold')
top_revenue = category_sales.sort_values('total_revenue', ascending=False).head(top_n).sort_values('total_revenue')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Subplot 1 – Total Terjual
bars1 = axes[0].barh(top_sold['product_category_name_english'], top_sold['total_sold'],
                     color=sns.color_palette('Blues_r', top_n))
axes[0].set_title(f'Top {top_n} Kategori Produk Terlaris', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Jumlah Terjual')

for bar in bars1:
    axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                 f'{int(bar.get_width()):,}', va='center', fontsize=9)

# Subplot 2 – Total Revenue
bars2 = axes[1].barh(top_revenue['product_category_name_english'], top_revenue['total_revenue'],
                     color=sns.color_palette('Greens_r', top_n))
axes[1].set_title(f'Top {top_n} Kategori Produk Revenue Tertinggi', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Total Revenue (BRL)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1e6:.1f}M'))

for bar in bars2:
    axes[1].text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
                 f'R${bar.get_width()/1e6:.2f}M', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# Insight:
# - Kategori bed_bath_table dan health_beauty mendominasi dari sisi jumlah produk terjual.
# - Dari sisi revenue, kategori health_beauty dan watches_gifts menghasilkan pendapatan tertinggi,
#   mengindikasikan harga produk yang lebih tinggi pada kategori tersebut.

# ============================================================
# Analisis Lanjutan (Opsional)
# ============================================================

# (Silakan tambahkan analisis geospatial atau analisis lanjutan lainnya di sini)

# ============================================================
# Conclusion
# ============================================================

# - Conclusion pertanyaan 1:
#   Tren jumlah pesanan bulanan menunjukkan pertumbuhan signifikan dari 2016 hingga 2018.
#   Puncak pesanan terjadi pada November 2017 dengan 7.289 pesanan, kemungkinan dipicu
#   oleh event promosi besar seperti Black Friday. Hal ini menunjukkan bahwa strategi
#   promosi musiman sangat berpengaruh terhadap volume penjualan.

# - Conclusion pertanyaan 2:
#   Kategori bed_bath_table adalah yang paling banyak terjual (10.953 unit), namun dari sisi
#   revenue, health_beauty menempati posisi tertinggi (R$1.23 juta). Ini menunjukkan bahwa
#   kategori dengan volume penjualan tertinggi tidak selalu menghasilkan revenue tertinggi,
#   dan ada peluang untuk meningkatkan margin pada kategori high-volume.

# - Conclusion pertanyaan 3:
#   Pelanggan terkonsentrasi di wilayah tenggara Brasil, terutama SP (Sao Paulo) dengan
#   41.746 pelanggan. Hal ini mencerminkan pusat ekonomi Brasil dan dapat menjadi acuan
#   untuk strategi distribusi dan pengiriman.

# - Conclusion pertanyaan 4:
#   Analisis RFM menunjukkan sebagian besar pelanggan berada di segmen 'At Risk' dan
#   'Potential Loyalists'. Frekuensi transaksi rata-rata hanya 1 kali, menandakan rendahnya
#   loyalitas pelanggan. Diperlukan strategi retensi seperti program loyalitas dan
#   personalized marketing untuk meningkatkan repeat order.
