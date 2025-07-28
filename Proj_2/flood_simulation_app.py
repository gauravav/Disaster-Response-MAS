import streamlit as st
import py3dep
import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for Streamlit
plt.switch_backend('Agg')

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
        st.info("Attempting to download elevation data...")

        # Method 1: Try using geometry directly
        try:
            dem_data = py3dep.get_dem(
                geometry=place_geom,
                resolution=30,  # 30m resolution
                crs="EPSG:4326"
            )
            elev_array = dem_data.values.squeeze()
            dem_bounds = dem_data.rio.bounds()
            extent = [dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]]
            st.success("✅ Method 1: Geometry-based download successful")

        except Exception as e1:
            st.warning(f"Method 1 failed: {e1}")
            st.info("Trying alternative method...")

            # Method 2: Use bounding box with py3depdem
            try:
                dem_data = py3dep.get_dem(
                    bbox=py3dep_bbox,
                    resolution=30,
                    crs="EPSG:4326"
                )
                elev_array = dem_data.values.squeeze()
                dem_bounds = dem_data.rio.bounds()
                extent = [dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]]
                st.success("✅ Method 2: Bbox-based download successful")

            except Exception as e2:
                st.warning(f"Method 2 failed: {e2}")
                st.info("Trying simplified approach...")

                # Method 3: Simple approach with basic parameters
                try:
                    bbox_simple = [bounds[1], bounds[0], bounds[3], bounds[2]]  # south, west, north, east
                    dem_data = py3dep.get_dem(bbox_simple, resolution=90)  # Lower resolution
                    elev_array = dem_data.squeeze()

                    # Create extent manually
                    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
                    st.success("✅ Method 3: Simple download successful")

                except Exception as e3:
                    st.error(f"All elevation download methods failed:")
                    st.error(f"Method 1: {e1}")
                    st.error(f"Method 2: {e2}")
                    st.error(f"Method 3: {e3}")
                    st.stop()

        # Validate elevation data
        if elev_array is None or elev_array.size == 0:
            st.error("Downloaded elevation data is empty")
            st.stop()

        # Handle NaN values
        nan_count = np.isnan(elev_array).sum()
        total_pixels = elev_array.size

        if nan_count == total_pixels:
            st.error("All elevation values are NaN - no valid data for this location")
            st.stop()
        elif nan_count > 0:
            st.warning(f"Found {nan_count}/{total_pixels} NaN values in elevation data")
            # Replace NaN with mean for visualization
            elev_array = np.where(np.isnan(elev_array), np.nanmean(elev_array), elev_array)

        # Display elevation statistics
        st.success(f"Elevation data: {elev_array.shape} pixels")
        st.info(f"Elevation range: {np.nanmin(elev_array):.1f}m to {np.nanmax(elev_array):.1f}m")
        st.info(f"Data extent: {extent}")

    except Exception as e:
        st.error(f"Critical error downloading elevation data: {e}")
        st.error("Please try a different location or check your internet connection")
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
with st.spinner("Creating visualization..."):
    try:
        # Create figure with explicit DPI and size
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
        plt.style.use('default')  # Use default matplotlib style

        # Check if elevation data is valid
        if elev_array.size == 0:
            st.error("No elevation data to display")
            st.stop()

        # Plot elevation as base layer with error handling
        try:
            elev_img = ax.imshow(
                elev_array,
                cmap='terrain',
                extent=extent,
                origin='upper',
                alpha=0.9,
                interpolation='bilinear'
            )

            # Add colorbar for elevation
            cbar = fig.colorbar(elev_img, ax=ax, label="Elevation (m)", shrink=0.8, pad=0.02)
            cbar.ax.tick_params(labelsize=10)

        except Exception as e:
            st.error(f"Error plotting elevation: {e}")
            st.write(f"Elevation array shape: {elev_array.shape}")
            st.write(f"Elevation extent: {extent}")
            st.write(f"Elevation data type: {elev_array.dtype}")
            st.stop()

        # Plot flood overlay
        if np.any(flood_mask):
            try:
                # Create masked array for flood areas
                flood_overlay = np.ma.masked_where(~flood_mask, np.ones_like(flood_mask))
                flood_img = ax.imshow(
                    flood_overlay,
                    cmap='Blues',
                    alpha=0.6,
                    extent=extent,
                    origin='upper',
                    vmin=0,
                    vmax=1
                )
            except Exception as e:
                st.warning(f"Error adding flood overlay: {e}")

        # Plot rivers
        if show_rivers and gdf_rivers is not None and not gdf_rivers.empty:
            try:
                gdf_rivers.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor='cyan',
                    linewidth=1.5,
                    alpha=0.8
                )
            except Exception as e:
                st.warning(f"Error plotting rivers: {e}")

        # Plot buildings
        if show_buildings and gdf_buildings is not None and not gdf_buildings.empty:
            try:
                gdf_buildings.plot(
                    ax=ax,
                    facecolor='red',
                    edgecolor='darkred',
                    linewidth=0.5,
                    alpha=0.7
                )
            except Exception as e:
                st.warning(f"Error plotting buildings: {e}")

        # Set labels and title
        ax.set_title(f"Flood Simulation: {place}\nWater Level: {water_level}m | Flooded Area: {flooded_area_pct:.1f}%",
                     fontsize=14, pad=20)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Set aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Add legend
        legend_elements = []
        if show_rivers and gdf_rivers is not None and not gdf_rivers.empty:
            legend_elements.append(plt.Line2D([0], [0], color='cyan', lw=2, label='Waterways'))
        if show_buildings and gdf_buildings is not None and not gdf_buildings.empty:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Buildings'))
        if np.any(flood_mask):
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.6, label='Flooded Area'))

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Adjust layout and display
        plt.tight_layout()

        # Display the plot
        st.pyplot(fig, clear_figure=True, use_container_width=True)

        # Close the figure to free memory
        plt.close(fig)

    except Exception as e:
        st.error(f"Critical error creating visualization: {e}")
        st.write("Debug information:")
        st.write(f"- Elevation array shape: {elev_array.shape if 'elev_array' in locals() else 'Not available'}")
        st.write(f"- Extent: {extent if 'extent' in locals() else 'Not available'}")
        st.write(f"- Matplotlib backend: {plt.get_backend()}")

        # Try a simple plot as fallback
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Map loading failed for {place}\nTry a different location",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title("Error Loading Map")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as fallback_error:
            st.error(f"Even fallback plot failed: {fallback_error}")

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