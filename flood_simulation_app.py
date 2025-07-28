import streamlit as st
import py3dep
import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import rasterio.transform
from rasterio.warp import transform_bounds

# -------------------------------
# Streamlit Setup
st.set_page_config(layout="wide")
st.title("🌊 Flood Simulation with Real Elevation Data (US Only)")

# -------------------------------
# User Inputs
place = st.sidebar.text_input("Enter US Location", "Dallas, Texas, USA")
water_level = st.sidebar.slider("Flood Water Level (m above sea level)", 0, 300, 100)
show_rivers = st.sidebar.checkbox("Overlay Rivers", True)
show_buildings = st.sidebar.checkbox("Overlay Buildings", False)

# -------------------------------
# Get bounding box
try:
    # Get the place geometry and bounds
    place_gdf = ox.geocode_to_gdf(place)
    place_geom = place_gdf.geometry[0]
    bounds = place_geom.bounds  # (minx, miny, maxx, maxy)

    # Create bbox in the format expected by different functions
    # For OSM: (north, south, east, west)
    osm_bbox = (bounds[3], bounds[1], bounds[2], bounds[0])
    # For py3dep: (south, west, north, east)
    py3dep_bbox = (bounds[1], bounds[0], bounds[3], bounds[2])

    st.success(f"Location: {place}")
    st.info(f"Bounds: West={bounds[0]:.4f}, South={bounds[1]:.4f}, East={bounds[2]:.4f}, North={bounds[3]:.4f}")

except Exception as e:
    st.error(f"Location not found or error geocoding: {e}")
    st.stop()

# -------------------------------
# Fetch elevation using py3dep
with st.spinner("Downloading elevation data..."):
    try:
        # Get DEM data with proper bbox format
        dem_data = py3dep.get_dem(
            geometry=place_geom,
            resolution=30,  # 30m resolution for better performance
            crs="EPSG:4326"
        )

        # Extract elevation array and transform
        elev_array = dem_data.values.squeeze()
        transform = dem_data.rio.transform()

        # Get the actual bounds of the DEM data
        dem_bounds = dem_data.rio.bounds()
        extent = [dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]]  # [west, east, south, north]

        st.success(f"Elevation data downloaded: {elev_array.shape} pixels")
        st.info(f"Elevation range: {np.nanmin(elev_array):.1f}m to {np.nanmax(elev_array):.1f}m")

    except Exception as e:
        st.error(f"Failed to download elevation data: {e}")
        st.stop()

# -------------------------------
# Flood Simulation
try:
    # Create flood mask (areas below water level)
    flood_mask = elev_array <= water_level
    flooded_area_pct = (np.sum(flood_mask) / flood_mask.size) * 100

    st.info(f"Flooded area: {flooded_area_pct:.1f}% of the region")

except Exception as e:
    st.error(f"Error in flood simulation: {e}")
    st.stop()

# -------------------------------
# Fetch OSM layers
gdf_rivers = None
gdf_buildings = None

if show_rivers or show_buildings:
    with st.spinner("Loading OpenStreetMap data..."):
        try:
            # Check OSMnx version and use appropriate function
            def get_osm_geometries(bbox, tags):
                """Get OSM geometries with version compatibility"""
                try:
                    # Try newer OSMnx version (>=1.0)
                    if hasattr(ox, 'geometries_from_bbox'):
                        return ox.geometries_from_bbox(*bbox, tags=tags)
                    # Try older OSMnx version
                    elif hasattr(ox, 'footprints_from_bbox'):
                        return ox.footprints_from_bbox(*bbox, footprint=tags)
                    # Even older version
                    elif hasattr(ox, 'pois_from_bbox'):
                        return ox.pois_from_bbox(*bbox, amenities=tags)
                    else:
                        raise Exception("OSMnx version not supported")
                except Exception as e:
                    # Fallback: try to get data using place name
                    if hasattr(ox, 'geometries_from_place'):
                        return ox.geometries_from_place(place, tags=tags)
                    else:
                        raise e

            # Fetch rivers/waterways
            if show_rivers:
                try:
                    gdf_rivers = get_osm_geometries(
                        osm_bbox,
                        {'waterway': ['river', 'stream', 'canal', 'ditch']}
                    )
                    if gdf_rivers is not None and not gdf_rivers.empty:
                        gdf_rivers = gdf_rivers.to_crs("EPSG:4326")
                        st.success(f"Found {len(gdf_rivers)} waterway features")
                    else:
                        st.info("No waterways found in this area")
                except Exception as e:
                    st.warning(f"Could not load rivers: {e}")

            # Fetch buildings
            if show_buildings:
                try:
                    gdf_buildings = get_osm_geometries(
                        osm_bbox,
                        {'building': True}
                    )
                    if gdf_buildings is not None and not gdf_buildings.empty:
                        gdf_buildings = gdf_buildings.to_crs("EPSG:4326")
                        st.success(f"Found {len(gdf_buildings)} building features")
                    else:
                        st.info("No buildings found in this area")
                except Exception as e:
                    st.warning(f"Could not load buildings: {e}")

        except Exception as e:
            st.warning(f"Error loading OSM data: {e}")

# -------------------------------
# Plotting
fig, ax = plt.subplots(figsize=(12, 10))

try:
    # Plot elevation as base layer
    elev_img = ax.imshow(
        elev_array,
        cmap='terrain',
        extent=extent,
        origin='upper',
        alpha=0.8
    )

    # Add colorbar for elevation
    cbar = fig.colorbar(elev_img, ax=ax, label="Elevation (m)", shrink=0.8)

    # Plot flood overlay
    if np.any(flood_mask):
        # Create masked array for flood areas
        flood_overlay = np.ma.masked_where(~flood_mask, np.ones_like(flood_mask))
        ax.imshow(
            flood_overlay,
            cmap='Blues',
            alpha=0.6,
            extent=extent,
            origin='upper',
            vmin=0,
            vmax=1
        )

    # Plot rivers
    if show_rivers and gdf_rivers is not None and not gdf_rivers.empty:
        gdf_rivers.plot(
            ax=ax,
            facecolor='none',
            edgecolor='cyan',
            linewidth=1.5,
            alpha=0.8
        )

    # Plot buildings
    if show_buildings and gdf_buildings is not None and not gdf_buildings.empty:
        gdf_buildings.plot(
            ax=ax,
            facecolor='red',
            edgecolor='darkred',
            linewidth=0.5,
            alpha=0.7
        )

    # Set labels and title
    ax.set_title(f"Flood Simulation: {place}\nWater Level: {water_level}m | Flooded Area: {flooded_area_pct:.1f}%",
                 fontsize=14, pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Set aspect ratio to equal for proper geographic display
    ax.set_aspect('equal')

    # Add legend
    legend_elements = []
    if show_rivers and gdf_rivers is not None and not gdf_rivers.empty:
        legend_elements.append(plt.Line2D([0], [0], color='cyan', lw=2, label='Waterways'))
    if show_buildings and gdf_buildings is not None and not gdf_buildings.empty:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Buildings'))
    if np.any(flood_mask):
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.6, label='Flooded Area'))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error creating visualization: {e}")

# -------------------------------
# Additional Information
st.subheader("ℹ️ Simulation Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Water Level", f"{water_level} m")

with col2:
    if 'flooded_area_pct' in locals():
        st.metric("Flooded Area", f"{flooded_area_pct:.1f}%")

with col3:
    if 'elev_array' in locals():
        st.metric("Elevation Range", f"{np.nanmin(elev_array):.0f}m - {np.nanmax(elev_array):.0f}m")

# Add disclaimer
st.warning("⚠️ This is a simplified flood simulation for educational purposes only. Real flood modeling requires additional factors like rainfall, drainage, soil permeability, and temporal dynamics.")