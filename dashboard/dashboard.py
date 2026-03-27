import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set(style='dark')

st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

@st.cache_data
def load_data():
    import os

    file_path = os.path.join(os.path.dirname(__file__), "main_data.csv")
    df = pd.read_csv(file_path)

    datetime_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])

    df['year_month_str'] = df['order_purchase_timestamp'].dt.strftime('%Y-%m')

    return df

df = load_data()

with st.sidebar:
    st.markdown("### Filter Rentang Waktu")

    min_date = df['order_purchase_timestamp'].min().date()
    max_date = df['order_purchase_timestamp'].max().date()

    start_date, end_date = st.date_input(
        label='Pilih Rentang Tanggal',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    st.markdown("---")
    st.markdown("**Hayqal Akbar Rizky Iskandar**")
    st.markdown("E-Commerce Public Dataset Analysis")

filtered = df[
    (df['order_purchase_timestamp'] >= pd.Timestamp(start_date)) &
    (df['order_purchase_timestamp'] <= pd.Timestamp(end_date))
].copy()

st.title("🛒 E-Commerce Public Dataset Dashboard")
st.markdown(f"Data periode: **{start_date}** s/d **{end_date}**")
st.markdown("---")

total_orders     = filtered['order_id'].nunique()
total_revenue    = filtered['payment_value'].sum()
unique_customers = filtered['customer_unique_id'].nunique()
avg_order_value  = filtered.drop_duplicates('order_id')['payment_value'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", f"{total_orders:,}")
with col2:
    st.metric("Total Revenue", format_currency(total_revenue, "BRL", locale='pt_BR'))
with col3:
    st.metric("Unique Customers", f"{unique_customers:,}")
with col4:
    st.metric("Avg Order Value", format_currency(avg_order_value, "BRL", locale='pt_BR'))

st.markdown("---")

st.subheader("📈 Tren Jumlah Pesanan Bulanan (2016–2018)")

monthly_orders = (
    filtered.drop_duplicates('order_id')
    .groupby('year_month_str')
    .agg(total_orders=('order_id', 'count'))
    .reset_index()
)

fig, ax = plt.subplots(figsize=(16, 5))
x = range(len(monthly_orders))
ax.plot(x, monthly_orders['total_orders'], marker='o', color='#1a73e8', linewidth=2.5, markersize=6)
ax.fill_between(x, monthly_orders['total_orders'], alpha=0.15, color='#1a73e8')

if len(monthly_orders) > 0:
    peak_idx = monthly_orders['total_orders'].idxmax()
    ax.annotate(
        f"Puncak: {monthly_orders.loc[peak_idx, 'total_orders']:,}\n{monthly_orders.loc[peak_idx, 'year_month_str']}",
        xy=(peak_idx, monthly_orders.loc[peak_idx, 'total_orders']),
        xytext=(max(0, peak_idx - 3), monthly_orders.loc[peak_idx, 'total_orders'] - 600),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=10, color='red', fontweight='bold'
    )

ax.set_xticks(x)
ax.set_xticklabels(monthly_orders['year_month_str'], rotation=45, ha='right', fontsize=9)
ax.set_title('Tren Jumlah Pesanan Bulanan', fontsize=14, fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Pesanan')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

st.subheader("🛍️ Kategori Produk Terlaris & Revenue Tertinggi")

category_sales = (
    filtered.groupby('product_category_name_english')
    .agg(total_sold=('order_id', 'count'), total_revenue=('price', 'sum'))
    .reset_index()
    .sort_values('total_sold', ascending=False)
)

top_n       = 10
top_sold    = category_sales.head(top_n).sort_values('total_sold')
top_revenue = category_sales.sort_values('total_revenue', ascending=False).head(top_n).sort_values('total_revenue')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

base_color      = '#A5D6A7'
highlight_color = '#2E7D32'

colors_sold = [highlight_color if i == len(top_sold) - 1 else base_color for i in range(len(top_sold))]
bars1 = axes[0].barh(top_sold['product_category_name_english'], top_sold['total_sold'], color=colors_sold)
axes[0].set_title(f'Top {top_n} Kategori Produk Terlaris', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Jumlah Terjual')
for bar in bars1:
    axes[0].text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                 f'{int(bar.get_width()):,}', va='center', fontsize=9)

colors_rev = [highlight_color if i == len(top_revenue) - 1 else base_color for i in range(len(top_revenue))]
bars2 = axes[1].barh(top_revenue['product_category_name_english'], top_revenue['total_revenue'], color=colors_rev)
axes[1].set_title(f'Top {top_n} Kategori Revenue Tertinggi', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Total Revenue (BRL)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'R${v/1e6:.1f}M'))
for bar in bars2:
    axes[1].text(bar.get_width() + 2000, bar.get_y() + bar.get_height()/2,
                 f'R${bar.get_width()/1e6:.2f}M', va='center', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

st.subheader("🗺️ Distribusi Geografis Pelanggan per State")

customer_state = (
    filtered.drop_duplicates('customer_unique_id')
    .groupby('customer_state')
    .agg(total_customers=('customer_unique_id', 'count'))
    .reset_index()
    .sort_values('total_customers', ascending=False)
)

fig, ax = plt.subplots(figsize=(14, 6))

base_color      = '#90CAF9'
highlight_color = '#1565C0'

colors = [highlight_color if i == 0 else base_color for i in range(len(customer_state))]
bars = ax.bar(customer_state['customer_state'], customer_state['total_customers'], color=colors)

for i, bar in enumerate(bars[:3]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{int(bar.get_height()):,}', ha='center', fontsize=8, fontweight='bold')

ax.set_title('Distribusi Pelanggan Berdasarkan State di Brasil', fontsize=13, fontweight='bold')
ax.set_xlabel('State')
ax.set_ylabel('Jumlah Pelanggan')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

st.subheader("🏅 RFM Analysis & Segmentasi Pelanggan")

reference_date = filtered['order_purchase_timestamp'].max()

rfm_df = (
    filtered.groupby('customer_unique_id')
    .agg(
        Recency=('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
        Frequency=('order_id', 'count'),
        Monetary=('payment_value', 'sum')
    )
    .reset_index()
)

rfm_df = rfm_df.dropna(subset=['Recency', 'Frequency', 'Monetary'])

rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], q=5, labels=False, duplicates='drop').apply(lambda x: 5 - x)
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=5, labels=False, duplicates='drop').apply(lambda x: x + 1)
rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], q=5, labels=False, duplicates='drop').apply(lambda x: x + 1)

rfm_df['RFM_Score'] = (
    rfm_df['R_Score'].astype(float).fillna(1) +
    rfm_df['F_Score'].astype(float).fillna(1) +
    rfm_df['M_Score'].astype(float).fillna(1)
).astype(int)

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

segment_order  = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Lost']
segment_colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#95a5a6']

segment_counts = (
    rfm_df['Segment']
    .value_counts()
    .reindex(segment_order)
    .fillna(0)
    .astype(int)
    .reset_index()
)
segment_counts.columns = ['Segment', 'Count']
segment_counts = segment_counts[segment_counts['Count'] > 0]

if segment_counts.empty:
    st.warning("Tidak ada data segmen untuk rentang tanggal ini.")
else:
    active_colors = [segment_colors[segment_order.index(s)] for s in segment_counts['Segment']]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bars = axes[0].bar(segment_counts['Segment'], segment_counts['Count'],
                       color=active_colors, edgecolor='white')
    axes[0].set_title('Jumlah Pelanggan per Segmen RFM', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Segmen')
    axes[0].set_ylabel('Jumlah Pelanggan')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{int(bar.get_height()):,}', ha='center', fontsize=9, fontweight='bold')

    axes[1].pie(
        segment_counts['Count'], labels=segment_counts['Segment'],
        autopct='%1.1f%%', colors=active_colors, startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    axes[1].set_title('Proporsi Segmen Pelanggan', fontsize=13, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

st.markdown("#### Distribusi Nilai Recency, Frequency, Monetary")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(rfm_df['Recency'],   ax=axes[0], color='#1a73e8', bins=30)
axes[0].set_title('Distribusi Recency',   fontsize=12, fontweight='bold')
axes[0].set_xlabel('Hari sejak transaksi terakhir')

sns.histplot(rfm_df['Frequency'], ax=axes[1], color='#f39c12', bins=30)
axes[1].set_title('Distribusi Frequency', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Jumlah Transaksi')

sns.histplot(rfm_df['Monetary'],  ax=axes[2], color='#2ecc71', bins=30)
axes[2].set_title('Distribusi Monetary',  fontsize=12, fontweight='bold')
axes[2].set_xlabel('Total Pembayaran (BRL)')

plt.tight_layout()
st.pyplot(fig)

st.markdown("#### 🏆 Top 10 Champions (Pelanggan Terbaik)")

top_champions = (
    rfm_df[rfm_df['Segment'] == 'Champions']
    [['customer_unique_id', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']]
    .sort_values('RFM_Score', ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top_champions.index += 1
st.dataframe(top_champions, use_container_width=True)

st.markdown("---")
st.caption("Hayqal Akbar Rizky Iskandar | Dicoding E-Commerce Public Dataset Analysis")