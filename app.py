# app.py
# Streamlit "premium" dashboard — Road Accident Intelligence (10 studi kasus)
# Jalankan: streamlit run app.py
#
# Catatan:
# - Letakkan road_accident_dataset.csv di folder yang sama dengan app.py (default).
# - App ini melakukan cleaning awal otomatis (typo severity, parsing tanggal/waktu, normalisasi kategori, dedup, dll).

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Road Accident Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme: dataset-representative (traffic-light accents + asphalt gradient)
# -----------------------------
SEVERITY_COLORS = {
    "Fatal": "#ff3b30",    # red
    "Serious": "#ff9f0a",  # amber
    "Slight": "#34c759",   # green
    "Unknown": "#9ca3af",  # neutral
}

CUSTOM_CSS = """
<style>
:root{
    --bg0:#070b14;
    --bg1:#0b1220;
    --card: rgba(255,255,255,.06);
    --card2: rgba(255,255,255,.08);
    --text: rgba(255,255,255,.92);
    --muted: rgba(255,255,255,.70);
    --muted2: rgba(255,255,255,.55);
    --shadow: 0 14px 36px rgba(0,0,0,.35);
    --radius: 18px;
}

/* App background */
[data-testid="stAppViewContainer"]{
    background:
      radial-gradient(900px 420px at 18% 0%, rgba(52,199,89,.18) 0%, transparent 55%),
      radial-gradient(900px 420px at 92% 12%, rgba(255,159,10,.16) 0%, transparent 55%),
      radial-gradient(900px 460px at 70% 92%, rgba(255,59,48,.12) 0%, transparent 60%),
      linear-gradient(180deg, var(--bg0), var(--bg1));
}

/* Main padding */
.block-container{ padding-top: 1.1rem; padding-bottom: 2.2rem; }
h1,h2,h3{ letter-spacing:-0.02em; color: var(--text); }
p,li,span,div{ color: var(--text); }

/* Sidebar (cleaner) */
[data-testid="stSidebar"]{
    background: rgba(255,255,255,.02);
    border-right: 1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebar"] .stMarkdown p{ color: var(--muted); }

/* Cards */
.card{
    border: 1px solid rgba(255,255,255,.12);
    border-radius: var(--radius);
    padding: 14px 16px;
    background: var(--card);
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
}
.card2{
    border: 1px solid rgba(255,255,255,.10);
    border-radius: var(--radius);
    padding: 16px 18px;
    background: var(--card2);
    box-shadow: 0 10px 28px rgba(0,0,0,.28);
    backdrop-filter: blur(10px);
}
.small{ font-size: .94rem; color: var(--muted); }

/* KPI */
.kpi{
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 18px;
    padding: 14px 14px 12px 14px;
    background: rgba(255,255,255,.06);
    box-shadow: 0 12px 30px rgba(0,0,0,.32);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}
.kpi:before{
    content: "";
    position:absolute; inset:0;
    background: linear-gradient(90deg, rgba(255,255,255,.10), transparent 60%);
    opacity:.55;
    pointer-events:none;
}
.kpi .label{ font-size:.82rem; color: var(--muted); margin-bottom:6px; }
.kpi .value{ font-size:1.62rem; font-weight:750; line-height:1.0; color: var(--text); }
.kpi .sub{ font-size:.78rem; color: var(--muted2); margin-top:6px; }

.kpi-accent{
    position:absolute; left:0; top:0; bottom:0; width:6px;
    border-radius: 18px 0 0 18px;
    opacity:.95;
}

hr{ border: 0; height: 1px; background: rgba(255,255,255,.08); margin: 0.9rem 0 1.2rem 0; }

/* Tables */
[data-testid="stDataFrame"]{
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.10);
}

/* Inputs */
div[data-baseweb="select"] > div{
    border-radius: 14px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

px.defaults.template = "plotly_dark"


# -----------------------------
# Helpers
# -----------------------------
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEVERITY_SCORE = {"Slight": 1, "Serious": 2, "Fatal": 3}


def _snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def _clean_category(x) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "Unknown"
    if "missing" in s.lower() or "out of range" in s.lower():
        return "Unknown"
    return s


def _fix_severity(x) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    s_low = s.lower()
    if s_low == "fetal":
        return "Fatal"
    s = s_low.capitalize()
    return s if s in {"Fatal", "Serious", "Slight"} else "Unknown"


def _style_plotly(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="rgba(255,255,255,0.88)",
        legend_title_text="",
        margin=dict(l=12, r=12, t=40, b=10),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.10)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.10)", zerolinecolor="rgba(255,255,255,0.12)")
    return fig


def kpi_card(label: str, value: str, sub: Optional[str] = None, accent: Optional[str] = None):
    accent_div = f'<div class="kpi-accent" style="background:{accent};"></div>' if accent else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi">
            {accent_div}
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def percent(numer: float, denom: float) -> float:
    return 0.0 if denom == 0 else 100.0 * numer / denom


@st.cache_data(show_spinner=False)
def load_and_clean(data_source) -> Tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(data_source)
    stats = {"rows_raw": int(len(raw)), "cols_raw": int(raw.shape[1])}

    df = raw.copy()
    df.columns = [_snake_case(c) for c in df.columns]

    # Dedup
    if "accident_index" in df.columns:
        dup = int(df.duplicated(subset=["accident_index"]).sum())
        stats["duplicates_accident_index"] = dup
        df = df.drop_duplicates(subset=["accident_index"], keep="first")
    else:
        stats["duplicates_accident_index"] = 0

    # Severity
    df["accident_severity"] = df["accident_severity"].map(_fix_severity)

    # Date/time
    df["accident_date"] = pd.to_datetime(df["accident_date"], format="%d-%m-%Y", errors="coerce")
    stats["date_parse_na"] = int(df["accident_date"].isna().sum())

    df["time_parsed"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce")
    df["hour"] = df["time_parsed"].dt.hour

    # Derived time features
    df["year"] = df["accident_date"].dt.year
    df["month"] = df["accident_date"].dt.to_period("M").astype(str)
    df["month_num"] = df["accident_date"].dt.month
    df["day_name"] = df["accident_date"].dt.day_name()

    # Day of week (prefer column, fallback to derived)
    df["day_of_week"] = df["day_of_week"].map(_clean_category)
    df.loc[df["day_of_week"].eq("Unknown"), "day_of_week"] = df["day_name"].fillna("Unknown")
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DOW_ORDER, ordered=True)

    # Categoricals
    cat_cols = [
        "junction_control",
        "junction_detail",
        "light_conditions",
        "local_authority_district",
        "carriageway_hazards",
        "police_force",
        "road_surface_conditions",
        "road_type",
        "urban_or_rural_area",
        "weather_conditions",
        "vehicle_type",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].map(_clean_category)

    # Numerics
    for c in ["number_of_casualties", "number_of_vehicles", "speed_limit", "latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["speed_limit"] = df["speed_limit"].round().astype("Int64")
    df["severity_score"] = df["accident_severity"].map(SEVERITY_SCORE).astype("Int64")

    # Keep carriageway hazards as Unknown rather than NaN
    if "carriageway_hazards" in df.columns:
        df["carriageway_hazards"] = df["carriageway_hazards"].fillna("Unknown")
        stats["missing_rate_carriageway_hazards"] = float(df["carriageway_hazards"].eq("Unknown").mean())
    else:
        stats["missing_rate_carriageway_hazards"] = 0.0

    stats["rows_clean"] = int(len(df))
    return df, stats


def apply_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Sidebar tanpa 'Control Panel' (biar nggak ada ruang kosong)."""
    with st.sidebar:
        st.markdown("## 🔎 Filter")
        with st.form("filters_form"):
            dmin = df["accident_date"].min()
            dmax = df["accident_date"].max()
            if pd.isna(dmin) or pd.isna(dmax):
                date_range = None
                st.info("Kolom tanggal tidak ter-parse. Filter tanggal dinonaktifkan.")
            else:
                date_range = st.date_input(
                    "Rentang tanggal",
                    value=(dmin.date(), dmax.date()),
                    min_value=dmin.date(),
                    max_value=dmax.date(),
                )

            sev = st.multiselect(
                "Accident Severity",
                options=["Fatal", "Serious", "Slight"],
                default=["Fatal", "Serious", "Slight"],
            )

            urban = st.multiselect(
                "Urban / Rural",
                options=sorted(df["urban_or_rural_area"].dropna().unique()),
                default=[],
                help="Kosongkan untuk semua.",
            )

            speed_opts = sorted(df["speed_limit"].dropna().unique().tolist())
            speed_sel = st.multiselect(
                "Speed limit",
                options=speed_opts,
                default=[],
                help="Kosongkan untuk semua.",
            )

            with st.expander("Filter lanjutan"):
                weather = st.multiselect(
                    "Weather conditions",
                    options=sorted(df["weather_conditions"].dropna().unique()),
                    default=[],
                )
                light = st.multiselect(
                    "Light conditions",
                    options=sorted(df["light_conditions"].dropna().unique()),
                    default=[],
                )
                road_type = st.multiselect(
                    "Road type",
                    options=sorted(df["road_type"].dropna().unique()),
                    default=[],
                )
                vehicle_type = st.multiselect(
                    "Vehicle type",
                    options=sorted(df["vehicle_type"].dropna().unique()),
                    default=[],
                )
                district = st.multiselect(
                    "Local authority (district)",
                    options=sorted(df["local_authority_district"].dropna().unique()),
                    default=[],
                )

            top_n_district = st.slider("Top-N district (ranking)", 5, 30, 15)
            st.form_submit_button("✅ Apply filters", use_container_width=True)

        st.markdown("---")
        st.markdown("## 🧪 Studi kasus")
        case = st.radio(
            "Pilih studi kasus",
            options=[
                "1) Tren waktu",
                "2) Komposisi severity",
                "3) Pola jam × hari",
                "4) Speed limit vs severity",
                "5) Cuaca vs severity",
                "6) Kondisi cahaya",
                "7) Road surface",
                "8) Vehicle type",
                "9) Urban vs Rural",
                "10) Hotspot district + peta",
            ],
            label_visibility="collapsed",
        )

    f = df.copy()

    if date_range:
        start, end = date_range
        f = f[(f["accident_date"] >= pd.Timestamp(start)) & (f["accident_date"] <= pd.Timestamp(end))]

    if sev:
        f = f[f["accident_severity"].isin(sev)]
    if urban:
        f = f[f["urban_or_rural_area"].isin(urban)]
    if speed_sel:
        f = f[f["speed_limit"].isin(speed_sel)]
    if district:
        f = f[f["local_authority_district"].isin(district)]
    if weather:
        f = f[f["weather_conditions"].isin(weather)]
    if light:
        f = f[f["light_conditions"].isin(light)]
    if road_type:
        f = f[f["road_type"].isin(road_type)]
    if vehicle_type:
        f = f[f["vehicle_type"].isin(vehicle_type)]

    selections = {
        "date_range": date_range,
        "severity": sev,
        "urban": urban,
        "speed_limit": speed_sel,
        "district": district,
        "top_n_district": top_n_district,
        "case": case,
    }
    return f, selections


# -----------------------------
# Data load (no uploader)
# -----------------------------
default_path = Path(__file__).parent / "road_accident_dataset.csv"
data_source = default_path

if not default_path.exists():
    st.error("File road_accident_dataset.csv tidak ditemukan di folder yang sama dengan app.py.")
    st.stop()

with st.spinner("Memuat & cleaning dataset..."):
    df, cleaning_stats = load_and_clean(data_source)

df_filt, selections = apply_filters(df)

# -----------------------------
# Hero header
# -----------------------------
date_min = df["accident_date"].min()
date_max = df["accident_date"].max()
date_range_txt = "—"
if not pd.isna(date_min) and not pd.isna(date_max):
    date_range_txt = f"{date_min.date()} → {date_max.date()}"

st.markdown(
    f"""
    <div class="card2">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="font-size:28px;">🚦</div>
            <div>
                <div style="font-size:40px; font-weight:800; line-height:1.05;">Road Accident Intelligence</div>
                <div class="small">10 studi kasus • visualisasi interaktif • tema “traffic risk” (warna merepresentasikan severity)</div>
            </div>
        </div>
        <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
            <span style="padding:6px 10px; border:1px solid rgba(255,255,255,.12); border-radius:999px; background:rgba(255,255,255,.04); color:rgba(255,255,255,.78);">Dataset range: {date_range_txt}</span>
            <span style="padding:6px 10px; border:1px solid rgba(255,255,255,.12); border-radius:999px; background:rgba(255,255,255,.04); color:rgba(255,255,255,.78);">Rows (clean): {cleaning_stats.get("rows_clean",0):,}</span>
            <span style="padding:6px 10px; border:1px solid rgba(255,255,255,.12); border-radius:999px; background:rgba(255,255,255,.04); color:rgba(255,255,255,.78);">Mode: {selections["case"]}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# -----------------------------
# KPI row
# -----------------------------
total = len(df_filt)
fatal = int((df_filt["accident_severity"] == "Fatal").sum())
serious = int((df_filt["accident_severity"] == "Serious").sum())
slight = int((df_filt["accident_severity"] == "Slight").sum())
avg_cas = float(df_filt["number_of_casualties"].mean()) if total else 0.0
avg_veh = float(df_filt["number_of_vehicles"].mean()) if total else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    kpi_card("Total kecelakaan", f"{total:,}", accent="linear-gradient(180deg,#60a5fa,#a78bfa)")
with k2:
    kpi_card("Fatal", f"{fatal:,}", f"{percent(fatal, total):.2f}% dari total", accent=SEVERITY_COLORS["Fatal"])
with k3:
    kpi_card("Serious", f"{serious:,}", f"{percent(serious, total):.2f}% dari total", accent=SEVERITY_COLORS["Serious"])
with k4:
    kpi_card("Slight", f"{slight:,}", f"{percent(slight, total):.2f}% dari total", accent=SEVERITY_COLORS["Slight"])
with k5:
    kpi_card("Avg casualties", f"{avg_cas:.2f}", accent="linear-gradient(180deg,#22c55e,#16a34a)")
with k6:
    kpi_card("Avg vehicles", f"{avg_veh:.2f}", accent="linear-gradient(180deg,#f59e0b,#ef4444)")

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Render selected case
# -----------------------------
case = selections["case"]


def _case_header(title: str, question: str):
    st.markdown(f"## {title}")
    st.markdown(f'<div class="card small"><b>Pertanyaan:</b> {question}</div>', unsafe_allow_html=True)
    st.write("")


if case == "1) Tren waktu":
    _case_header("1) Tren kecelakaan dari waktu ke waktu", "kapan tren kecelakaan naik/turun? ada pola musiman?")

    ts = (
        df_filt.dropna(subset=["accident_date"])
        .assign(month=lambda d: d["accident_date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "accidents"})
    )
    if len(ts) == 0:
        st.info("Tidak ada data tanggal untuk divisualisasikan pada filter saat ini.")
    else:
        ts["month_dt"] = pd.to_datetime(ts["month"] + "-01", errors="coerce")
        ts = ts.sort_values("month_dt")
        ts["ma3"] = ts["accidents"].rolling(3, min_periods=1).mean()

        fig = px.line(
            ts,
            x="month_dt",
            y=["accidents", "ma3"],
            markers=True,
            labels={"value": "Jumlah kecelakaan", "month_dt": "Bulan", "variable": "Series"},
        )
        st.plotly_chart(_style_plotly(fig), use_container_width=True)

elif case == "2) Komposisi severity":
    _case_header("2) Komposisi Accident Severity", "seberapa besar porsi Fatal/Serious/Slight?")

    sev_counts = (
        df_filt["accident_severity"]
        .value_counts()
        .reindex(["Fatal", "Serious", "Slight"], fill_value=0)
        .reset_index()
    )
    sev_counts.columns = ["accident_severity", "count"]

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(
            sev_counts,
            x="accident_severity",
            y="count",
            text="count",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            labels={"accident_severity": "Severity", "count": "Jumlah"},
        )
        st.plotly_chart(_style_plotly(fig), use_container_width=True)
    with c2:
        fig = px.pie(
            sev_counts,
            names="accident_severity",
            values="count",
            hole=0.45,
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(_style_plotly(fig), use_container_width=True)

elif case == "3) Pola jam × hari":
    _case_header("3) Pola kecelakaan berdasarkan Jam × Hari (Heatmap)", "jam dan hari mana yang paling rawan?")

    hm = df_filt.dropna(subset=["hour"]).copy()
    if len(hm) == 0:
        st.info("Tidak ada data jam (time) untuk divisualisasikan pada filter saat ini.")
    else:
        heat = (
            hm.groupby(["day_of_week", "hour"], as_index=False)
            .size()
            .rename(columns={"size": "accidents"})
        )
        fig = px.density_heatmap(
            heat,
            x="hour",
            y="day_of_week",
            z="accidents",
            nbinsx=24,
            labels={"hour": "Jam (0-23)", "day_of_week": "Hari", "accidents": "Jumlah"},
        )
        fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": DOW_ORDER})
        st.plotly_chart(_style_plotly(fig), use_container_width=True)

elif case == "4) Speed limit vs severity":
    _case_header("4) Dampak Speed Limit terhadap Severity", "apakah fatal/serious meningkat di speed limit tinggi?")

    g = (
        df_filt.dropna(subset=["speed_limit"])
        .groupby(["speed_limit", "accident_severity"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if len(g) == 0:
        st.info("Tidak ada data speed_limit pada filter saat ini.")
    else:
        fig = px.bar(
            g,
            x="speed_limit",
            y="count",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"speed_limit": "Speed limit", "count": "Jumlah", "accident_severity": "Severity"},
        )
        st.plotly_chart(_style_plotly(fig), use_container_width=True)

        pivot = g.pivot_table(index="speed_limit", columns="accident_severity", values="count", aggfunc="sum", fill_value=0)
        pivot["total"] = pivot.sum(axis=1)
        pivot["severe_rate_%"] = 100 * (pivot.get("Fatal", 0) + pivot.get("Serious", 0)) / pivot["total"]
        pivot = pivot.reset_index().sort_values("speed_limit")

        fig2 = px.line(pivot, x="speed_limit", y="severe_rate_%", markers=True,
                       labels={"speed_limit": "Speed limit", "severe_rate_%": "Severe rate (%)"})
        st.plotly_chart(_style_plotly(fig2), use_container_width=True)
        st.caption("Severe rate = (Fatal + Serious) / Total, per speed limit.")

elif case == "5) Cuaca vs severity":
    _case_header("5) Weather Conditions vs Severity", "cuaca apa yang paling sering & bagaimana severity-nya?")

    top_weather_n = st.slider("Top-N weather ditampilkan", 5, 20, 10, key="top_weather")
    w = df_filt.copy()
    w["weather_conditions"] = w["weather_conditions"].fillna("Unknown")
    top_weather = w["weather_conditions"].value_counts().head(top_weather_n).index.tolist()
    w = w[w["weather_conditions"].isin(top_weather)]

    g = (
        w.groupby(["weather_conditions", "accident_severity"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if len(g) == 0:
        st.info("Tidak ada data cuaca pada filter saat ini.")
    else:
        fig = px.bar(
            g,
            x="weather_conditions",
            y="count",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"weather_conditions": "Cuaca", "count": "Jumlah"},
        )
        fig.update_layout(xaxis={"tickangle": -18})
        st.plotly_chart(_style_plotly(fig), use_container_width=True)

elif case == "6) Kondisi cahaya":
    _case_header("6) Light Conditions & Jam", "kondisi cahaya apa dominan? jam berapa lebih rawan?")

    c1, c2 = st.columns([1, 1])
    with c1:
        lc = (
            df_filt["light_conditions"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "light_conditions", "light_conditions": "count"})
        )
        lc.columns = ["light_conditions", "count"]
        fig = px.bar(lc, x="light_conditions", y="count", labels={"light_conditions": "Light", "count": "Jumlah"})
        fig.update_layout(xaxis={"tickangle": -15})
        st.plotly_chart(_style_plotly(fig), use_container_width=True)

    with c2:
        hh = (
            df_filt.dropna(subset=["hour"])
            .groupby(["hour", "light_conditions"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        if len(hh) == 0:
            st.info("Tidak ada data jam untuk visualisasi kedua.")
        else:
            fig = px.line(hh, x="hour", y="count", color="light_conditions", markers=True,
                          labels={"hour": "Jam", "count": "Jumlah"})
            st.plotly_chart(_style_plotly(fig), use_container_width=True)

elif case == "7) Road surface":
    _case_header("7) Road Surface Conditions & Severity", "permukaan jalan (dry/wet/ice) mempengaruhi severity?")

    rs = (
        df_filt.groupby(["road_surface_conditions", "accident_severity"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if len(rs) == 0:
        st.info("Tidak ada data road_surface_conditions pada filter saat ini.")
    else:
        fig = px.bar(
            rs,
            x="road_surface_conditions",
            y="count",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"road_surface_conditions": "Road surface", "count": "Jumlah"},
        )
        fig.update_layout(xaxis={"tickangle": -18})
        st.plotly_chart(_style_plotly(fig), use_container_width=True)

        pv = rs.pivot_table(index="road_surface_conditions", columns="accident_severity", values="count", aggfunc="sum", fill_value=0)
        pv["total"] = pv.sum(axis=1)
        pv["severe_rate_%"] = 100 * (pv.get("Fatal", 0) + pv.get("Serious", 0)) / pv["total"]
        pv = pv.reset_index().sort_values("severe_rate_%", ascending=False)

        fig2 = px.bar(pv, x="road_surface_conditions", y="severe_rate_%", labels={"severe_rate_%": "Severe rate (%)"})
        fig2.update_layout(xaxis={"tickangle": -18})
        st.plotly_chart(_style_plotly(fig2), use_container_width=True)

elif case == "8) Vehicle type":
    _case_header("8) Vehicle Type", "tipe kendaraan mana paling sering & bagaimana severity-nya?")

    top_vehicle_n = st.slider("Top-N vehicle ditampilkan", 5, 25, 12, key="top_vehicle")
    v = df_filt.copy()
    top_vehicle = v["vehicle_type"].value_counts().head(top_vehicle_n).index.tolist()
    v = v[v["vehicle_type"].isin(top_vehicle)]

    g = (
        v.groupby(["vehicle_type", "accident_severity"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if len(g) == 0:
        st.info("Tidak ada data vehicle_type pada filter saat ini.")
    else:
        fig = px.bar(
            g,
            x="vehicle_type",
            y="count",
            color="accident_severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"vehicle_type": "Vehicle type", "count": "Jumlah"},
        )
        fig.update_layout(xaxis={"tickangle": -18})
        st.plotly_chart(_style_plotly(fig), use_container_width=True)

elif case == "9) Urban vs Rural":
    _case_header("9) Urban vs Rural", "urban atau rural lebih banyak? bagaimana severe rate-nya?")

    ur = (
        df_filt.groupby(["urban_or_rural_area", "accident_severity"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if len(ur) == 0:
        st.info("Tidak ada data urban_or_rural_area pada filter saat ini.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(
                ur,
                x="urban_or_rural_area",
                y="count",
                color="accident_severity",
                color_discrete_map=SEVERITY_COLORS,
                barmode="stack",
                labels={"urban_or_rural_area": "Area", "count": "Jumlah"},
            )
            st.plotly_chart(_style_plotly(fig), use_container_width=True)

        with c2:
            pv = ur.pivot_table(index="urban_or_rural_area", columns="accident_severity", values="count", aggfunc="sum", fill_value=0)
            pv["total"] = pv.sum(axis=1)
            pv["severe_rate_%"] = 100 * (pv.get("Fatal", 0) + pv.get("Serious", 0)) / pv["total"]
            pv = pv.reset_index().sort_values("severe_rate_%", ascending=False)
            fig2 = px.bar(pv, x="urban_or_rural_area", y="severe_rate_%", labels={"severe_rate_%": "Severe rate (%)", "urban_or_rural_area": "Area"})
            st.plotly_chart(_style_plotly(fig2), use_container_width=True)

elif case == "10) Hotspot district + peta":
    _case_header("10) Hotspot district + peta", "district mana paling rawan? dimana cluster lokasinya?")

    top_n = selections.get("top_n_district", 15)

    d = df_filt.dropna(subset=["local_authority_district"]).copy()
    d["is_severe"] = d["accident_severity"].isin(["Fatal", "Serious"]).astype(int)

    rank = (
        d.groupby("local_authority_district", as_index=False)
        .agg(
            accidents=("accident_index", "count"),
            severe_rate=("is_severe", "mean"),
            avg_severity=("severity_score", "mean"),
            avg_casualties=("number_of_casualties", "mean"),
        )
    )
    rank["severe_rate_%"] = 100 * rank["severe_rate"]
    rank = rank.sort_values(["accidents"], ascending=False)

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("### Ranking berdasarkan jumlah kecelakaan")
        fig = px.bar(
            rank.head(top_n),
            x="accidents",
            y="local_authority_district",
            orientation="h",
            labels={"accidents": "Jumlah", "local_authority_district": "District"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(_style_plotly(fig), use_container_width=True)

    with c2:
        st.markdown("### Ranking berdasarkan severe rate (%)")
        rank2 = rank.sort_values("severe_rate_%", ascending=False)
        fig2 = px.bar(
            rank2.head(top_n),
            x="severe_rate_%",
            y="local_authority_district",
            orientation="h",
            labels={"severe_rate_%": "Severe rate (%)", "local_authority_district": "District"},
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(_style_plotly(fig2), use_container_width=True)

    st.markdown("### Peta hotspot (weighted by severity)")
    geo = df_filt.dropna(subset=["latitude", "longitude"]).copy()
    if len(geo) == 0:
        st.info("Tidak ada data koordinat untuk peta pada filter saat ini.")
    else:
        geo["severity_w"] = geo["severity_score"].fillna(1).astype(float)
        max_points = 150_000
        if len(geo) > max_points:
            geo = geo.sample(max_points, random_state=42)

        view_state = pdk.ViewState(
            latitude=float(geo["latitude"].mean()),
            longitude=float(geo["longitude"].mean()),
            zoom=5.4,
            pitch=45,
        )

        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=geo,
            get_position="[longitude, latitude]",
            radius=650,
            elevation_scale=35,
            elevation_range=[0, 4000],
            pickable=True,
            extruded=True,
            get_elevation_weight="severity_w",
            elevation_aggregation="SUM",
            get_color_weight="severity_w",
            color_aggregation="MEAN",
        )

        # Use a free public style (no Mapbox token needed)
        carto_dark = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

        deck = pdk.Deck(
            layers=[hex_layer],
            initial_view_state=view_state,
            map_style=carto_dark,
            tooltip={"text": "Hex hotspot (weighted by severity)"},
        )
        st.pydeck_chart(deck, use_container_width=True)

    st.markdown("### Tabel ringkas (top 50)")
    show_cols = ["local_authority_district", "accidents", "severe_rate_%", "avg_severity", "avg_casualties"]
    st.dataframe(rank[show_cols].head(50), use_container_width=True, hide_index=True)

# -----------------------------
# Export
# -----------------------------

