import streamlit as st
import redis
import pandas as pd
import plotly.express as px
from datetime import datetime

# Initialize Redis
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    st.success("🟢 Redis connected")
except Exception as e:
    st.error(f"🔴 Redis connection failed: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="Confirmed Flood Zones", page_icon="🌊")
st.title("🌊 Confirmed Flood Zones Map")
st.markdown("This dashboard shows the final predictions from both **sensor_agent** and **tweets_agent** combined by **coordination_agent**.")

# Fetch confirmed flood zones from Redis
confirmed = redis_client.xrevrange("confirmed_flood_zones", max='+', min='-', count=100)

if not confirmed:
    st.info("No confirmed flood zones yet.")
else:
    data = []
    for entry_id, fields in confirmed:
        data.append({
            "Timestamp": fields.get("timestamp"),
            "Latitude": float(fields.get("lat", 0)),
            "Longitude": float(fields.get("lon", 0)),
            "Source": fields.get("confirmed_by", "N/A")
        })

    df = pd.DataFrame(data)

    # Show map
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Source",
        size_max=15,
        zoom=10,
        hover_name="Timestamp",
        height=700,
        title="🗺️ Confirmed Flood Prediction Locations"
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Raw Data Table"):
        st.dataframe(df)

# Optional: Auto-refresh every 10 seconds
st.caption("🔄 Refreshes every 10 seconds")
st.rerun()
