import time
from io import BytesIO

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

REFRESH_S = 1.0
REQUEST_TIMEOUT_S = 0.8
LOGO_PATH = "output/web/logo.svg"
HEARTBEAT_TIMEOUT_S = 2.0

st.set_page_config(
    page_title="Toplink Vision", layout="wide", initial_sidebar_state="collapsed"
)

st_autorefresh(interval=int(REFRESH_S * 1000), key="auto_refresh")

st.markdown(
    """
    <style>
      .block-container { padding: 0.3rem 0.8rem; }
      div.stButton > button { width: 110px; }
      header, footer { visibility: hidden; height: 0; }
      #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_status(base_url: str):
    try:
        r = requests.get(f"{base_url}/status", timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def fetch_preview(base_url: str):
    try:
        r = requests.get(f"{base_url}/preview/latest", timeout=REQUEST_TIMEOUT_S)
        if r.status_code != 200:
            return None, None
        return r.content, r.headers.get("Content-Type", "image/jpeg")
    except Exception:
        return None, None


def post_trigger(base_url: str):
    try:
        requests.post(f"{base_url}/trigger", timeout=REQUEST_TIMEOUT_S)
    except Exception:
        return False
    return True


def result_color(text: str) -> str:
    t = str(text).upper()
    if t == "OK":
        return "#2ecc71"
    if t == "NG":
        return "#ff4d4d"
    if t in ("ERR", "ERROR", "TIMEOUT"):
        return "#ffb020"
    return "#ffffff"


def load_inline_logo() -> str:
    try:
        with open(LOGO_PATH, "r", encoding="utf-8") as f:
            svg = f.read()
    except Exception:
        return ""
    svg = svg.replace('fill="#ffffff"', 'fill="currentColor"')
    svg = svg.replace('fill="#fff"', 'fill="currentColor"')
    if "currentColor" not in svg:
        svg = svg.replace("<svg ", '<svg fill="currentColor" ')
    return svg


def html_escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def is_online(last_tick_ts: float | None) -> bool:
    if last_tick_ts is None:
        return False
    return (time.time() - float(last_tick_ts)) <= HEARTBEAT_TIMEOUT_S


def get_records(status_data: dict | None):
    if status_data:
        return status_data.get("records") or []
    return []


def get_latest_record(status_data: dict | None):
    records = get_records(status_data)
    return records[0] if records else None


def get_stats(status_data: dict | None):
    if status_data:
        return status_data.get("stats") or {}
    return {}


def get_last_tick(status_data: dict | None):
    if status_data:
        return status_data.get("last_tick_ts")
    return None


def format_trigger_dt(value: str) -> tuple[str, str]:
    return value[:10], value[11:23]


def build_table_html(records: list[dict]) -> str:
    rows_html = []
    for rec in records:
        trig = rec.get("triggered_at", "")
        date, time_str = format_trigger_dt(trig)
        res = str(rec.get("result", "")).upper()
        color = "#2ecc71" if res == "OK" else ("#ff4d4d" if res == "NG" else "#ffb020")
        ms_val = f"{float(rec.get('duration_ms', 0.0)):.1f}"
        rows_html.append(
            "<tr>"
            f"<td>{html_escape(str(rec.get('trigger_seq', 0)).rjust(5, '0'))}</td>"
            f"<td style='color:{color}'>{html_escape(rec.get('result', ''))}</td>"
            f"<td>{html_escape(date)}</td>"
            f"<td>{html_escape(time_str)}</td>"
            f"<td>{html_escape(ms_val)}</td>"
            "</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:4px 6px;border-bottom:1px solid #ddd;'>ID</th>"
        "<th style='text-align:left;padding:4px 6px;border-bottom:1px solid #ddd;'>Result</th>"
        "<th style='text-align:left;padding:4px 6px;border-bottom:1px solid #ddd;'>Date</th>"
        "<th style='text-align:left;padding:4px 6px;border-bottom:1px solid #ddd;'>Time</th>"
        "<th style='text-align:left;padding:4px 6px;border-bottom:1px solid #ddd;'>ms</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody></table>"
    )


base_url = "http://127.0.0.1:8000"

status_data, status_err = fetch_status(base_url)
latest_result = ""
latest_record = get_latest_record(status_data)
if latest_record:
    latest_result = latest_record.get("result", "")
title_color = result_color(latest_result)
inline_logo = load_inline_logo()

left, right = st.columns([7, 3], gap="medium")

with left:
    if status_data is not None:
        img_bytes, img_mime = fetch_preview(base_url)
        if img_bytes:
            st.image(BytesIO(img_bytes), width="stretch")
        else:
            st.info("No preview available")
    else:
        st.warning("Status unreachable")

with right:
    # Header panel: logo + title on left, clock on right
    header_left, header_right = st.columns([3, 2], gap="small")
    with header_left:
        logo_html = inline_logo.replace(
            "<svg",
            f"<svg style='color:{title_color};height:40px;width:auto;vertical-align:middle;margin-right:8px;'",
        )
        st.markdown(
            f"<div style='display:flex;align-items:center;color:{title_color};font-size:1.6rem;font-weight:600;'>{logo_html}<span>Toplink Vision</span></div>",
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown("### " + time.strftime("%H:%M:%S"))

    st.markdown("")

    # Status + Trigger row
    status_col, trigger_col = st.columns([3, 2], gap="small")
    with status_col:
        if status_data is None:
            st.write("Status unreachable")
        else:
            latest = get_latest_record(status_data)
            if latest:
                latest_id = str(latest.get("trigger_seq", 0)).rjust(5, "_")
                latest_dt = latest.get("triggered_at", "")
                latest_line = f"<span style='color:{title_color}'>{latest_id} {latest_dt.replace('T', ' ').replace('Z', '')}</span>"
                st.markdown(latest_line, unsafe_allow_html=True)
            else:
                st.write("Idle")
    with trigger_col:
        if st.button("Trigger", use_container_width=False):
            post_trigger(base_url)

    st.markdown("")

    # Stats row
    if status_data is None:
        st.info("No stats")
    else:
        stats = get_stats(status_data)
        col1, col2, col3, col4, col5 = st.columns(5, gap="small")
        total = stats.get("total", 0)
        ok = stats.get("ok", 0)
        ng = stats.get("ng", 0)
        err = stats.get("error", 0)
        pass_rate = stats.get("pass_rate", 0.0) * 100
        col1.markdown(
            f"**Total**<br><span style='font-size:22px;'>{total}</span>",
            unsafe_allow_html=True,
        )
        col2.markdown(
            f"**OK**<br><span style='color:#2ecc71;font-size:22px;'>{ok}</span>",
            unsafe_allow_html=True,
        )
        col3.markdown(
            f"**NG**<br><span style='color:#ff4d4d;font-size:22px;'>{ng}</span>",
            unsafe_allow_html=True,
        )
        col4.markdown(
            f"**Err**<br><span style='color:#ffb020;font-size:22px;'>{err}</span>",
            unsafe_allow_html=True,
        )
        col5.markdown(
            f"**Pass%**<br><span style='font-size:22px;'>{pass_rate:.1f}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Records table
    if status_data is None:
        st.info("No records")
    else:
        records = get_records(status_data)
        if records:
            st.markdown(build_table_html(records), unsafe_allow_html=True)
        else:
            st.info("No records")

    last_tick = get_last_tick(status_data)
    online = is_online(last_tick)
    dot_color = "#2ecc71" if online else "#ff4d4d"
    status_text = "Runtime: Online" if online else "Runtime: Offline"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-top:8px;'>"
        f"<span style='width:10px;height:10px;border-radius:50%;background:{dot_color};display:inline-block;'></span>"
        f"<span>{status_text}</span></div>",
        unsafe_allow_html=True,
    )
