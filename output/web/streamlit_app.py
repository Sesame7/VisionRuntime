import time
from io import BytesIO

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

REFRESH_S = 1.0
REQUEST_TIMEOUT_S = 0.8
LOGO_PATH = "output/web/logo.svg"
HEARTBEAT_STALE_POLLS = 3

st.set_page_config(
    page_title="Toplink Vision", layout="wide", initial_sidebar_state="collapsed"
)

st_autorefresh(interval=int(REFRESH_S * 1000), key="auto_refresh")

st.markdown(
    """
    <style>
      .block-container { padding: 0.3rem 0.8rem; }
      div[data-testid="stButton"] { margin: 0; width: 100%; display: flex; justify-content: flex-end; }
      div[data-testid="stButton"] button { width: 110px; height: 38px; margin: 0; }
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
        return r.json()
    except Exception:
        return None


def fetch_preview(base_url: str):
    try:
        r = requests.get(f"{base_url}/preview/latest", timeout=REQUEST_TIMEOUT_S)
        if r.status_code != 200:
            return None
        return r.content
    except Exception:
        return None


def post_trigger(base_url: str):
    try:
        requests.post(f"{base_url}/trigger", timeout=REQUEST_TIMEOUT_S)
    except Exception:
        return


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


def get_heartbeat_seq(status_data: dict | None):
    if status_data:
        return status_data.get("heartbeat_seq")
    return None


def _init_heartbeat_state():
    if "hb_last_seq" not in st.session_state:
        st.session_state.hb_last_seq = None
    if "hb_has_progress" not in st.session_state:
        st.session_state.hb_has_progress = False
    if "hb_stale_polls" not in st.session_state:
        st.session_state.hb_stale_polls = HEARTBEAT_STALE_POLLS


def heartbeat_status(status_data: dict | None) -> str:
    _init_heartbeat_state()
    heartbeat_seq = get_heartbeat_seq(status_data)

    if not isinstance(heartbeat_seq, int):
        st.session_state.hb_stale_polls += 1
        alive = (
            st.session_state.hb_has_progress
            and st.session_state.hb_stale_polls < HEARTBEAT_STALE_POLLS
        )
        return "ok" if alive else "ng"

    if st.session_state.hb_last_seq is None:
        st.session_state.hb_last_seq = heartbeat_seq
        st.session_state.hb_stale_polls = 0
        return "pending"

    if heartbeat_seq > st.session_state.hb_last_seq:
        st.session_state.hb_last_seq = heartbeat_seq
        st.session_state.hb_has_progress = True
        st.session_state.hb_stale_polls = 0
        return "ok"

    if heartbeat_seq < st.session_state.hb_last_seq:
        st.session_state.hb_last_seq = heartbeat_seq
        st.session_state.hb_has_progress = False
        st.session_state.hb_stale_polls = 0
        return "pending"

    st.session_state.hb_stale_polls += 1
    alive = (
        st.session_state.hb_has_progress
        and st.session_state.hb_stale_polls < HEARTBEAT_STALE_POLLS
    )
    return "ok" if alive else "ng"


def format_trigger_dt(value: str) -> tuple[str, str]:
    return value[:10], value[11:24]


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


def build_stat_html(
    label: str, value: str | int | float, color: str | None = None
) -> str:
    value_style = "font-size:22px;font-weight:600;line-height:1.05;margin-top:2px;"
    if color:
        value_style += f"color:{color};"
    return (
        "<div style='display:flex;flex-direction:column;justify-content:center;"
        "line-height:1.05;margin:0;padding:0;'>"
        f"<div style='font-size:0.9rem;color:#888;line-height:1;margin:0;padding:0;'>{html_escape(str(label))}</div>"
        f"<div style='{value_style}'>{html_escape(str(value))}</div>"
        "</div>"
    )


base_url = "http://127.0.0.1:8000"

status_data = fetch_status(base_url)
latest_result = ""
latest_record = get_latest_record(status_data)
if latest_record:
    latest_result = latest_record.get("result", "")
title_color = result_color(latest_result)
inline_logo = load_inline_logo()

left, right = st.columns([7, 3], gap="medium")

with left:
    if status_data is not None:
        img_bytes = fetch_preview(base_url)
        if img_bytes:
            st.image(BytesIO(img_bytes), width="stretch")
        else:
            st.info("No preview available")
    else:
        st.warning("Status unreachable")

with right:
    # Header panel: keep title and clock on one row with vertical centering.
    logo_html = inline_logo.replace(
        "<svg",
        f"<svg style='color:{title_color};height:40px;width:auto;vertical-align:middle;margin-right:8px;'",
    )
    st.markdown(
        f"""
        <div style='display:flex;align-items:center;justify-content:space-between;gap:12px;'>
          <div style='display:flex;align-items:center;color:{title_color};font-size:1.6rem;font-weight:600;min-width:0;'>
            {logo_html}<span>Toplink Vision</span>
          </div>
          <div style='font-size:1.5rem;font-weight:600;line-height:1;white-space:nowrap;'>{time.strftime("%H:%M:%S")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    # Status + Trigger row
    status_col, trigger_col = st.columns([3, 2], gap="small")
    with status_col:
        if status_data is None:
            st.markdown(
                "<div style='display:flex;align-items:center;min-height:38px;'>Status unreachable</div>",
                unsafe_allow_html=True,
            )
        else:
            latest = get_latest_record(status_data)
            if latest:
                latest_id = str(latest.get("trigger_seq", 0)).rjust(5, "_")
                latest_dt = latest.get("triggered_at", "")
                latest_line = (
                    "<div style='display:flex;align-items:center;min-height:38px;'>"
                    f"<span style='color:{title_color};line-height:1;'>{latest_id} {latest_dt.replace('T', ' ')}</span>"
                    "</div>"
                )
                st.markdown(latest_line, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='display:flex;align-items:center;min-height:38px;'>Idle</div>",
                    unsafe_allow_html=True,
                )
    with trigger_col:
        if st.button("Trigger", use_container_width=False):
            post_trigger(base_url)

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

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
            build_stat_html("Total", total),
            unsafe_allow_html=True,
        )
        col2.markdown(
            build_stat_html("OK", ok, "#2ecc71"),
            unsafe_allow_html=True,
        )
        col3.markdown(
            build_stat_html("NG", ng, "#ff4d4d"),
            unsafe_allow_html=True,
        )
        col4.markdown(
            build_stat_html("Err", err, "#ffb020"),
            unsafe_allow_html=True,
        )
        col5.markdown(
            build_stat_html("Pass%", f"{pass_rate:.1f}"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    # Records table
    if status_data is None:
        st.info("No records")
    else:
        records = get_records(status_data)
        if records:
            st.markdown(build_table_html(records), unsafe_allow_html=True)
        else:
            st.info("No records")

    runtime_state = heartbeat_status(status_data)
    if runtime_state == "ok":
        dot_color = "#2ecc71"
        status_text = "Runtime: Online"
    elif runtime_state == "pending":
        dot_color = "#777777"
        status_text = "Runtime: Connecting..."
    else:
        dot_color = "#ff4d4d"
        status_text = "Runtime: Offline"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-top:8px;'>"
        f"<span style='width:10px;height:10px;border-radius:50%;background:{dot_color};display:inline-block;'></span>"
        f"<span>{status_text}</span></div>",
        unsafe_allow_html=True,
    )
