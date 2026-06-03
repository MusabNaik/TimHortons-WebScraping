"""
Tim Hortons Canada — Interactive Data Dashboard
Streamlit app showcasing data analysis, geospatial intelligence,
and ML-powered insights from 4,000+ locations across Canada.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium, folium_static
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tim Hortons Canada — Data Analysis",
    page_icon="🍩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800; color: #C8102E; margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem; color: #666; margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center;
        border-left: 4px solid #C8102E;
    }
    .metric-card h3 { margin: 0; color: #333; font-size: 0.9rem; }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: #C8102E; }
    .insight-box {
        background: #e8f4f8; border-radius: 8px; padding: 12px 15px;
        border-left: 4px solid #2b6cb0; margin: 10px 0;
    }
    .insight-box strong { color: #2b6cb0; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading 4,000+ Tim Hortons locations...")
def load_data():
    df = pd.read_csv("Tim_Hortons_Locations.csv")

    # Clean column name typo
    df.rename(columns={
        "Provience": "Province",
        "Has WiFi": "WiFi",
        "Catering Available": "Catering",
        "Dine In": "DineIn",
        "Take Out": "TakeOut"
    }, inplace=True)

    # Clean whitespace
    for col in ["Province", "cities"]:
        df[col] = df[col].str.strip()

    # Convert Yes/No to boolean
    for col in ["WiFi", "Catering", "DineIn", "TakeOut"]:
        df[col] = df[col].map({"Yes": True, "No": False, "yes": True, "no": False})

    # Standardize province names (handle bilingual/fr)
    province_map = {
        "Alberta": "AB", "British Columbia": "BC", "Ontario": "ON",
        "Quebec": "QC", "Saskatchewan": "SK", "Manitoba": "MB",
        "Nova Scotia": "NS", "New Brunswick": "NB", "Newfoundland and Labrador": "NL",
        "Prince Edward Island": "PE", "Nunavut": "NU", "Northwest Territories": "NT",
        "Yukon": "YT", "Nunavut / Nunavut": "NU"
    }
    df["ProvinceCode"] = df["Province"].map(province_map).fillna(df["Province"])

    # Fix missing lat/lng rows
    df = df[(df["lat"] != 0) & (df["lng"] != 0)].copy()

    # Parse hours
    hour_cols = [c for c in df.columns if "Dine-In" in c or "Drive-Thru" in c]
    for col in hour_cols:
        df[col] = df[col].astype(str).str.strip()

    # Feature count
    df["AmenityCount"] = df[["WiFi", "Catering", "DineIn", "TakeOut"]].sum(axis=1)

    # Is 24h (Dine-In Midnight - 11:59 PM = open 24h)
    df["DineIn24h"] = df["Dine-In Monday"].str.contains("Midnight", na=False)

    # Has drive-thru (at least one Drive-Thru column not empty)
    dt_cols = [c for c in df.columns if "Drive-Thru" in c]
    df["HasDriveThru"] = df[dt_cols].apply(
        lambda r: any(v not in ["", "Closed", "nan"] for v in r), axis=1
    )

    return df

df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color:#C8102E;'>🍩 Filters</h2>", unsafe_allow_html=True)

provinces = sorted(df["Province"].unique())
selected_provinces = st.sidebar.multiselect(
    "Province(s)", provinces, default=provinces
)

amenity_filter = st.sidebar.multiselect(
    "Must have amenities",
    ["WiFi", "Catering", "DineIn", "TakeOut", "HasDriveThru"],
    default=[]
)

search_city = st.sidebar.text_input("🔍 Search city", placeholder="e.g., Toronto")

# Apply filters
mask = df["Province"].isin(selected_provinces)
for a in amenity_filter:
    mask &= df[a] == True

if search_city:
    mask &= df["cities"].str.lower().str.contains(search_city.lower())

filtered = df[mask].copy()

# Stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**📍 {len(filtered):,}** locations shown")
st.sidebar.markdown(f"**🏛️ {filtered['Province'].nunique()}** provinces")
st.sidebar.markdown(f"**🏙️ {filtered['cities'].nunique()}** cities")

# ── Header ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">🍩 Tim Hortons Canada</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Data Analysis · Geospatial Intelligence · ML Clustering</p>',
                unsafe_allow_html=True)
with col2:
    total = len(df)
    shown = len(filtered)
    st.markdown(
        f"<div class='metric-card'><h3>Total Locations</h3>"
        f"<div class='value'>{total:,}</div></div>",
        unsafe_allow_html=True
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🗺️ Interactive Map", "☕ Amenities",
    "🕐 Hours Analysis", "📍 Geospatial & ML", "🏪 Store Typology"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📍 Nationwide Presence")

    # Top-level KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"<div class='metric-card'><h3>📍 Locations</h3>"
            f"<div class='value'>{len(filtered):,}</div></div>",
            unsafe_allow_html=True
        )
    with k2:
        pct_wifi = (filtered["WiFi"].sum() / len(filtered) * 100) if len(filtered) else 0
        st.markdown(
            f"<div class='metric-card'><h3>📶 WiFi Available</h3>"
            f"<div class='value'>{pct_wifi:.0f}%</div></div>",
            unsafe_allow_html=True
        )
    with k3:
        pct_dt = (filtered["HasDriveThru"].sum() / len(filtered) * 100) if len(filtered) else 0
        st.markdown(
            f"<div class='metric-card'><h3>🚗 Has Drive-Thru</h3>"
            f"<div class='value'>{pct_dt:.0f}%</div></div>",
            unsafe_allow_html=True
        )
    with k4:
        pct_24h = (filtered["DineIn24h"].sum() / len(filtered) * 100) if len(filtered) else 0
        st.markdown(
            f"<div class='metric-card'><h3>🌙 24h Dine-In</h3>"
            f"<div class='value'>{pct_24h:.0f}%</div></div>",
            unsafe_allow_html=True
        )
    with k5:
        pct_cater = (filtered["Catering"].sum() / len(filtered) * 100) if len(filtered) else 0
        st.markdown(
            f"<div class='metric-card'><h3>🍽️ Catering</h3>"
            f"<div class='value'>{pct_cater:.0f}%</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Province distribution
    col_a, col_b = st.columns(2)

    with col_a:
        prov_counts = filtered["Province"].value_counts().reset_index()
        prov_counts.columns = ["Province", "Count"]
        fig = px.bar(
            prov_counts, x="Province", y="Count",
            color="Count", color_continuous_scale="Reds",
            title="Locations by Province",
            text_auto=True, height=400
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Top cities
        city_counts = filtered["cities"].value_counts().head(15).reset_index()
        city_counts.columns = ["City", "Count"]
        fig = px.bar(
            city_counts, x="Count", y="City",
            orientation="h", color="Count", color_continuous_scale="Reds",
            title="Top 15 Cities by Location Count",
            text_auto=True, height=400
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Key insight boxes
    top_city = filtered["cities"].value_counts().index[0]
    top_prov = filtered["Province"].value_counts().index[0]
    most_amenities = filtered.groupby("Province")[["WiFi", "Catering", "DineIn", "TakeOut"]].mean().idxmax()
    highest_wifi_prov = most_amenities["WiFi"]

    st.markdown(f"""
    <div class='insight-box'>
        <strong>🔑 Key Insights</strong><br>
        • <strong>{top_city}</strong> has the most Tim Hortons locations in the selection<br>
        • <strong>{top_prov}</strong> leads with the highest total location count<br>
        • <strong>{pct_wifi:.0f}%</strong> of locations offer free WiFi — an essential amenity metric<br>
        • <strong>{pct_dt:.0f}%</strong> have a Drive-Thru, showcasing the car-centric nature of the brand<br>
        • <strong>{pct_24h:.0f}%</strong> operate 24-hour Dine-In service
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — INTERACTIVE MAP
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🗺️ Interactive Store Locator Map")

    col_c, col_d = st.columns([3, 1])
    with col_d:
        map_style = st.selectbox(
            "Map style", ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark"],
            index=1
        )
        color_by = st.selectbox(
            "Colour markers by",
            ["WiFi", "Catering", "DineIn", "TakeOut", "HasDriveThru", "AmenityCount"],
            index=0
        )
        show_clusters = st.checkbox("Show clustering", value=True)

    with col_c:
        # Don't try to render more than 2000 points
        map_df = filtered.dropna(subset=["lat", "lng"]).copy()
        if len(map_df) > 2000:
            map_df = map_df.sample(2000, random_state=42)
            st.caption(f"Showing 2,000 sampled locations ({len(filtered):,} total — sampling for performance)")

        if not map_df.empty:
            tile_map = {
                "OpenStreetMap": "OpenStreetMap",
                "CartoDB Positron": "CartoDB positron",
                "CartoDB Dark": "CartoDB dark_matter"
            }

            center_lat = map_df["lat"].mean()
            center_lng = map_df["lng"].mean()

            m = folium.Map(
                location=[center_lat, center_lng],
                zoom_start=4,
                tiles=tile_map.get(map_style, "OpenStreetMap"),
                control_scale=True,
                zoom_control=True
            )

            if color_by == "AmenityCount":
                color_map = {0: "gray", 1: "orange", 2: "green", 3: "blue", 4: "purple"}
            else:
                color_map = {True: "#2ecc71", False: "#e74c3c"}

            marker_group = folium.FeatureGroup(name="Locations")

            for _, row in map_df.iterrows():
                if color_by == "AmenityCount":
                    color = color_map.get(int(row["AmenityCount"]), "gray")
                else:
                    color = color_map.get(row.get(color_by, False), "gray")

                popup_html = f"""
                <div style="font-family: Arial, sans-serif; min-width:250px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/b/be/Tim_Hortons_Logo.svg"
                         style="width:50%; display:block; margin:0 auto;">
                    <hr>
                    <strong>{row['address']}</strong><br>
                    {row['cities']}, {row['ProvinceCode']}<br><br>
                    📶 {'✅ WiFi' if row['WiFi'] else '❌ No WiFi'} ·
                    🚗 {'✅ Drive-Thru' if row.get('HasDriveThru', False) else '❌ No Drive-Thru'}<br>
                    🍽️ {'✅ Catering' if row['Catering'] else '❌ No Catering'} ·
                    🪑 {'✅ Dine-In' if row['DineIn'] else '❌ No Dine-In'}
                </div>
                """
                folium.Marker(
                    location=[row["lat"], row["lng"]],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color="red" if color == "#e74c3c" else "green",
                                     icon="info-sign")
                ).add_to(marker_group)

            if show_clusters and len(map_df) > 50:
                from folium.plugins import MarkerCluster
                marker_cluster = MarkerCluster().add_to(m)
                for _, row in map_df.iterrows():
                    folium.Marker(
                        location=[row["lat"], row["lng"]],
                        popup=row["address"][:50] if len(str(row["address"])) > 50 else row["address"]
                    ).add_to(marker_cluster)

            if not show_clusters:
                marker_group.add_to(m)

            folium.LayerControl().add_to(m)

            st_folium(m, width=None, height=550)
        else:
            st.warning("No locations match the current filter selection.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — AMENITIES ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("☕ Amenities & Features Analysis")

    col_e, col_f = st.columns(2)

    with col_e:
        # Amenities donut chart
        amenities_data = {
            "Amenity": ["WiFi", "Drive-Thru", "Dine-In", "Take-Out", "Catering"],
            "Available": [
                filtered["WiFi"].sum(),
                filtered["HasDriveThru"].sum(),
                filtered["DineIn"].sum(),
                filtered["TakeOut"].sum(),
                filtered["Catering"].sum()
            ],
            "Not Available": [
                len(filtered) - filtered["WiFi"].sum(),
                len(filtered) - filtered["HasDriveThru"].sum(),
                len(filtered) - filtered["DineIn"].sum(),
                len(filtered) - filtered["TakeOut"].sum(),
                len(filtered) - filtered["Catering"].sum()
            ]
        }
        adf = pd.DataFrame(amenities_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=adf["Amenity"], x=adf["Available"],
            name="Available", orientation="h",
            marker_color="#C8102E", text=adf["Available"], textposition="outside"
        ))
        fig.add_trace(go.Bar(
            y=adf["Amenity"], x=adf["Not Available"],
            name="Not Available", orientation="h",
            marker_color="#e0e0e0", text=adf["Not Available"], textposition="outside"
        ))
        fig.update_layout(
            title="Amenity Availability (All Locations)",
            barmode="stack", height=350, hovermode="y unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        # Amenities by province heatmap
        prov_amenities = filtered.groupby("Province")[
            ["WiFi", "Catering", "DineIn", "TakeOut", "HasDriveThru"]
        ].mean().mul(100).round(1)

        fig = px.imshow(
            prov_amenities,
            text_auto=True, aspect="auto",
            color_continuous_scale="Reds",
            title="Amenity Availability by Province (%)",
            height=400
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    # Amenity correlation heatmap
    st.markdown("---")
    corr_cols = ["WiFi", "Catering", "DineIn", "TakeOut", "HasDriveThru", "AmenityCount"]
    corr_df = filtered[corr_cols].corr()

    fig = px.imshow(
        corr_df, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Amenity Correlation Matrix — What tends to co-occur?",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
        <strong>💡 Correlation Insights</strong><br>
        • A high correlation between <strong>WiFi and Dine-In</strong> suggests dining customers expect connectivity<br>
        • <strong>Drive-Thru</strong> shows moderate negative correlation with Dine-In — highway locations vs sit-down<br>
        • <strong>Catering</strong> is the rarest amenity, typically offered at larger, full-service locations
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — HOURS ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🕐 Operating Hours Analysis")

    # Extract open/close times from Dine-In Monday
    def extract_hours(hour_str):
        """Parse '5:00 AM - 10:00 PM' or 'Closed' or 'Midnight - 11:59 PM'"""
        if pd.isna(hour_str) or str(hour_str).strip() in ["", "nan", "Closed"]:
            return None, None
        hour_str = str(hour_str).strip()
        if "Midnight" in hour_str:
            return 0.0, 23.99  # open 24h
        parts = re.split(r'\s*-\s*', hour_str)
        if len(parts) != 2:
            return None, None

        def to_24h(t):
            t = t.strip()
            if "Midnight" in t:
                return 0.0
            match = re.match(r'(\d+):(\d+)\s*(AM|PM)', t, re.I)
            if not match:
                return None
            h, m, ap = int(match.group(1)), int(match.group(2)), match.group(3).upper()
            if ap == "PM" and h != 12:
                h += 12
            if ap == "AM" and h == 12:
                h = 0
            return h + m / 60.0

        open_t = to_24h(parts[0])
        close_t = to_24h(parts[1])
        return open_t, close_t

    # Calculate for Dine-In Monday
    hours_data = filtered["Dine-In Monday"].apply(extract_hours)
    filtered = filtered.copy()
    filtered["OpenHour"] = hours_data.apply(lambda x: x[0])
    filtered["CloseHour"] = hours_data.apply(lambda x: x[1])
    filtered["HoursOpen"] = filtered.apply(
        lambda r: (r["CloseHour"] - r["OpenHour"]) if (r["OpenHour"] is not None and r["CloseHour"] is not None) else None,
        axis=1
    )

    col_g, col_h = st.columns(2)

    with col_g:
        # Distribution of opening hours
        open_hrs = filtered["OpenHour"].dropna()
        fig = px.histogram(
            open_hrs, nbins=24,
            title="When Do Locations Open? (Dine-In Monday)",
            labels={"value": "Hour (24h)", "count": "# of Locations"},
            color_discrete_sequence=["#C8102E"],
            height=350
        )
        fig.add_vline(x=5, line_dash="dash", line_color="green",
                      annotation_text="5 AM (typical)", annotation_position="top right")
        fig.add_vline(x=6, line_dash="dash", line_color="blue",
                      annotation_text="6 AM", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

    with col_h:
        # Distribution of closing hours
        close_hrs = filtered["CloseHour"].dropna()
        fig = px.histogram(
            close_hrs, nbins=24,
            title="When Do Locations Close? (Dine-In Monday)",
            labels={"value": "Hour (24h)", "count": "# of Locations"},
            color_discrete_sequence=["#2b6cb0"],
            height=350
        )
        fig.add_vline(x=22, line_dash="dash", line_color="orange",
                      annotation_text="10 PM (typical)", annotation_position="top left")
        fig.add_vline(x=23.99, line_dash="dash", line_color="green",
                      annotation_text="24h", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)

    col_i, col_j = st.columns(2)

    with col_i:
        # Hours open distribution
        hours_open = filtered["HoursOpen"].dropna()
        fig = px.histogram(
            hours_open, nbins=20,
            title="How Many Hours Are Locations Open?",
            labels={"value": "Hours Open", "count": "# of Locations"},
            color_discrete_sequence=["#8b5cf6"],
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_j:
        # Sunday closures
        sunday_closed = (filtered["Dine-In Sunday"].astype(str).str.strip() == "Closed").sum()
        sunday_open = len(filtered) - sunday_closed
        fig = go.Figure(data=[go.Pie(
            labels=["Open Sunday", "Closed Sunday"],
            values=[sunday_open, sunday_closed],
            marker_colors=["#2ecc71", "#e74c3c"],
            hole=0.5,
            textinfo="label+percent"
        )])
        fig.update_layout(title="Sunday Dine-In Operation", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # 24h locations map
    st.markdown("---")
    st.subheader("🌙 24-Hour Locations")
    _24h = filtered[filtered["DineIn24h"] == True]
    if len(_24h) > 0:
        st.markdown(f"**{len(_24h):,}** locations ({len(_24h)/len(filtered)*100:.1f}%) operate 24-hour Dine-In")

        col_k, col_l = st.columns(2)
        with col_k:
            _24h_prov = _24h["Province"].value_counts().head(10).reset_index()
            _24h_prov.columns = ["Province", "Count"]
            fig = px.bar(_24h_prov, x="Province", y="Count",
                         color="Count", color_continuous_scale="Reds",
                         title="24h Locations by Province",
                         text_auto=True, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col_l:
            # Drive-Thru hours summary
            dt_sun = filtered["Drive-Thru Sunday"].value_counts().head(5).reset_index()
            dt_sun.columns = ["Hours", "Count"]
            fig = px.pie(dt_sun, values="Count", names="Hours",
                         title="Drive-Thru Sunday Hours Distribution",
                         hole=0.4, height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 24-hour locations in current selection.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — GEOSPATIAL & ML
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📍 Geospatial Analysis & Machine Learning")

    cluster_method = st.selectbox(
        "Clustering method",
        ["K-Means (market regions)", "DBSCAN (density-based)"],
        index=0
    )

    coords = filtered[["lat", "lng"]].dropna().values
    if len(coords) < 10:
        st.warning("Not enough data points for clustering.")
    else:
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)

        if "K-Means" in cluster_method:
            n_clusters = st.slider("Number of clusters (K)", 2, 10, 5)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(coords_scaled)
            cluster_centers = scaler.inverse_transform(km.cluster_centers_)

            # Calculate inertia for elbow
            inertias = []
            k_range = range(1, 11)
            for k in k_range:
                if k == 1:
                    inertias.append(None)
                    continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(coords_scaled)
                inertias.append(kmeans.inertia_)

        else:  # DBSCAN
            eps = st.slider("Epsilon (neighbourhood radius)", 0.1, 3.0, 0.5, 0.1)
            min_samples = st.slider("Min samples per cluster", 3, 30, 10)
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(coords_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_centers = None
            inertias = None

        col_m, col_n = st.columns(2)

        with col_m:
            # Map of clusters
            cluster_df = filtered.iloc[:len(labels)].copy()
            cluster_df["Cluster"] = labels.astype(str)

            center_lat = cluster_df["lat"].mean()
            center_lng = cluster_df["lng"].mean()
            m = folium.Map(location=[center_lat, center_lng], zoom_start=4,
                           tiles="CartoDB positron")

            cluster_colors = px.colors.qualitative.Set2
            for _, row in cluster_df.iterrows():
                cl = row["Cluster"]
                color = cluster_colors[int(cl) % len(cluster_colors)] if cl != "-1" else "#888"
                folium.CircleMarker(
                    location=[row["lat"], row["lng"]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"Cluster {cl}"
                ).add_to(m)

            if cluster_centers is not None:
                for i, (lat, lng) in enumerate(cluster_centers):
                    folium.Marker(
                        location=[lat, lng],
                        icon=folium.Icon(color="black", icon="star", prefix="fa"),
                        popup=f"<b>Cluster {i} Center</b>"
                    ).add_to(m)

            st_folium(m, width=None, height=450)

        with col_n:
            # Cluster composition by province
            if n_clusters > 0 and n_clusters < 20:
                prov_cluster = pd.crosstab(
                    cluster_df["Province"], cluster_df["Cluster"]
                )
                fig = px.imshow(
                    prov_cluster, text_auto=True, aspect="auto",
                    color_continuous_scale="Reds",
                    title=f"Province Distribution by Cluster ({n_clusters} clusters)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        # Elbow / noise info
        col_o, col_p = st.columns(2)
        with col_o:
            if inertias:
                valid_k = [k for k, v in zip(range(1, 11), inertias) if v is not None]
                valid_i = [v for v in inertias if v is not None]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(valid_k), y=valid_i,
                    mode="lines+markers",
                    marker=dict(size=8, color="#C8102E"),
                    line=dict(color="#C8102E", width=2)
                ))
                fig.add_vline(x=n_clusters, line_dash="dash", line_color="green",
                              annotation_text=f"Selected: K={n_clusters}")
                fig.update_layout(
                    title="Elbow Method — Optimal K",
                    xaxis_title="K (clusters)", yaxis_title="Inertia",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_p:
            noise_count = (labels == -1).sum()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>📍 Clusters Found</h3>
                <div class='value'>{n_clusters}</div>
            </div>
            """, unsafe_allow_html=True)
            if "DBSCAN" in cluster_method and noise_count > 0:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>⚠️ Noise Points</h3>
                    <div class='value' style='color:#e74c3c;'>{noise_count:,}</div>
                </div>
                """, unsafe_allow_html=True)

    # Nearest neighbour analysis
    st.markdown("---")
    st.subheader("📏 Store Proximity Analysis")

    if len(coords) >= 100:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        nearest_dist = distances[:, 1]  # distance to nearest neighbour (not self)

        # Convert to km (rough: 1 degree ≈ 111km at Canada's latitude ~50°N)
        deg_to_km = 111.0
        nearest_km = nearest_dist * deg_to_km

        col_q, col_r = st.columns(2)
        with col_q:
            fig = px.histogram(
                nearest_km, nbins=30,
                title="Distance to Nearest Tim Hortons (km)",
                labels={"value": "Distance (km)", "count": "# of Locations"},
                color_discrete_sequence=["#C8102E"],
                height=350
            )
            fig.add_vline(x=nearest_km.mean(), line_dash="dash", line_color="green",
                          annotation_text=f"Mean: {nearest_km.mean():.1f} km")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            avg_dist_prov = filtered.iloc[:len(nearest_km)].copy()
            avg_dist_prov["NearestKM"] = nearest_km
            prov_dist = avg_dist_prov.groupby("Province")["NearestKM"].mean().sort_values().reset_index()
            prov_dist.columns = ["Province", "Avg Nearest (km)"]
            fig = px.bar(
                prov_dist, x="Province", y="Avg Nearest (km)",
                color="Avg Nearest (km)", color_continuous_scale="Reds",
                title="Avg Distance to Nearest Store by Province",
                text_auto=".1f", height=350
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class='insight-box'>
            <strong>📍 Proximity Insight</strong><br>
            The average distance between Tim Hortons locations is <strong>{nearest_km.mean():.1f} km</strong>.
            Dense urban markets (ON, QC) show much tighter spacing than rural provinces.
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — STORE TYPOLOGY
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🏪 Store Type Classification")

    st.markdown("""
    Using K-Means clustering on amenity profiles to discover natural store archetypes.
    Each location is classified by its combination of features (WiFi, Catering, Drive-Thru, etc.)
    + operating hours profile.
    """)

    # Feature engineering for store typology
    typology_df = filtered.copy()

    # Encode amenities as numeric features
    for col in ["WiFi", "Catering", "DineIn", "TakeOut", "HasDriveThru", "DineIn24h"]:
        typology_df[col + "_num"] = typology_df[col].astype(int)

    # Encode province
    le = LabelEncoder()
    typology_df["ProvinceCode_num"] = le.fit_transform(typology_df["ProvinceCode"])

    # Hours features (normalize to 0-1)
    typology_df["Open_Norm"] = (typology_df["OpenHour"] - typology_df["OpenHour"].min()) / (
        typology_df["OpenHour"].max() - typology_df["OpenHour"].min() + 0.001
    ) if typology_df["OpenHour"].notna().any() else 0
    typology_df["Close_Norm"] = (typology_df["CloseHour"] - typology_df["CloseHour"].min()) / (
        typology_df["CloseHour"].max() - typology_df["CloseHour"].min() + 0.001
    ) if typology_df["CloseHour"].notna().any() else 0
    typology_df["HoursOpen_Norm"] = (typology_df["HoursOpen"] - typology_df["HoursOpen"].min()) / (
        typology_df["HoursOpen"].max() - typology_df["HoursOpen"].min() + 0.001
    ) if typology_df["HoursOpen"].notna().any() else 0

    # Sunday closed flag
    typology_df["SundayClosed"] = (typology_df["Dine-In Sunday"].astype(str).str.strip() == "Closed").astype(int)

    # Feature matrix
    feature_cols = [
        "WiFi_num", "Catering_num", "DineIn_num", "TakeOut_num",
        "HasDriveThru_num", "DineIn24h_num",
        "Open_Norm", "Close_Norm", "HoursOpen_Norm", "SundayClosed"
    ]
    X = typology_df[feature_cols].dropna()

    if len(X) < 50:
        st.warning("Not enough data for classification.")
    else:
        n_types = st.slider("Number of store types (clusters)", 2, 6, 3)
        km_types = KMeans(n_clusters=n_types, random_state=42, n_init=10)
        type_labels = km_types.fit_predict(X)

        type_df = X.copy()
        type_df["Type"] = type_labels.astype(str)

        # Analyze each type
        type_profile = type_df.groupby("Type")[
            ["WiFi_num", "Catering_num", "DineIn_num", "TakeOut_num",
             "HasDriveThru_num", "DineIn24h_num", "SundayClosed"]
        ].mean()

        # Rename for display
        type_profile.columns = ["WiFi", "Catering", "Dine-In", "Take-Out",
                                "Drive-Thru", "24h", "Sunday Closed"]
        type_profile = type_profile.mul(100).round(1)

        # Assign descriptive names
        type_names = {}
        for t in type_profile.index:
            row = type_profile.loc[t]
            if row["Drive-Thru"] > 70 and row["Dine-In"] < 40:
                name = "🚗 Drive-Thru Focused"
            elif row["Drive-Thru"] > 70 and row["Dine-In"] > 70:
                name = "🔄 Full Service + Drive-Thru"
            elif row["24h"] > 60:
                name = "🌙 24-Hour Operation"
            elif row["Dine-In"] > 70 and row["Drive-Thru"] < 30:
                name = "🪑 Dine-In Focused"
            elif row["Catering"] > 40:
                name = "🍽️ Catering Hub"
            elif row["Sunday Closed"] > 50:
                name = "📅 Weekday Only"
            elif row["WiFi"] > 80 and row["Dine-In"] < 40:
                name = "📶 WiFi Cafe Style"
            else:
                name = "🏪 Standard Location"
            type_names[t] = name

        type_profile.index = [type_names[t] for t in type_profile.index]

        # Heatmap of type profiles
        fig = px.imshow(
            type_profile, text_auto=".0f", aspect="auto",
            color_continuous_scale="Reds",
            title="Store Type Profiles — % of Locations with Each Feature",
            height=300 + 40 * n_types
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribution of types
        type_dist = type_df["Type"].value_counts().reset_index()
        type_dist.columns = ["Type", "Count"]
        type_dist["Label"] = type_dist["Type"].map(type_names)
        type_dist = type_dist.sort_values("Type")

        fig = px.pie(
            type_dist, values="Count", names="Label",
            title="Distribution of Store Types",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Which provinces have which types?
        type_df_full = typology_df.loc[X.index].copy()
        type_df_full["Type"] = type_labels
        type_df_full["TypeName"] = type_df_full["Type"].astype(str).map(type_names)

        prov_type = pd.crosstab(type_df_full["Province"], type_df_full["TypeName"])
        fig = px.imshow(
            prov_type, text_auto=True, aspect="auto",
            color_continuous_scale="Reds",
            title="Store Type Distribution by Province",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class='insight-box'>
            <strong>🏪 Typology Insight</strong><br>
            The analysis reveals <strong>{n_types} distinct store archetypes</strong> across Canada.
            Provinces like Ontario and Alberta show a diverse mix of types, while smaller provinces
            are dominated by standard full-service locations.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>"
    "🍩 <strong>Tim Hortons Canada Location Intelligence</strong> · "
    f"Data from {total:,} locations across {df['Province'].nunique()} provinces · "
    "Built with Streamlit + Folium + Scikit-Learn"
    "</p>",
    unsafe_allow_html=True
)
