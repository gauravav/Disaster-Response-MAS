import redis
import json
import pandas as pd
from datetime import datetime
import time
import os
import threading
from collections import defaultdict, deque
import statistics

class LiveSensorMonitor:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize the live sensor monitor"""
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.connected = True
            print(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            print("💡 Make sure Redis is running: redis-server")
            self.connected = False
            return

        self.stream_name = "sensor_data"
        self.is_monitoring = False
        self.sensor_data_history = defaultdict(lambda: deque(maxlen=100))  # Keep last 100 readings per sensor
        self.last_readings = {}
        self.alert_counts = defaultdict(int)

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def format_sensor_reading(self, sensor_data):
        """Format a single sensor reading for display"""
        sensor_id = sensor_data.get('sensor_id', 'Unknown')
        sensor_type = sensor_data.get('sensor_type', 'Unknown')
        reading = float(sensor_data.get('current_reading', 0))
        water_depth = float(sensor_data.get('water_depth', 0))
        alert_level = sensor_data.get('alert_level', 'normal')
        status = sensor_data.get('status', 'unknown')
        is_flooded = sensor_data.get('is_flooded', 'False') == 'True'
        timestamp = sensor_data.get('timestamp', '')
        lat = float(sensor_data.get('lat', 0))
        lon = float(sensor_data.get('lon', 0))
        elevation = float(sensor_data.get('elevation', 0))

        # Color coding for alert levels
        colors = {
            'normal': '\033[92m',    # Green
            'caution': '\033[93m',   # Yellow
            'warning': '\033[91m',   # Red
            'critical': '\033[95m'   # Magenta
        }
        reset_color = '\033[0m'
        color = colors.get(alert_level, '\033[0m')

        # Status icons
        flood_icon = "🌊" if is_flooded else "🏔️"
        alert_icon = {
            'normal': '🟢',
            'caution': '🟡',
            'warning': '🟠',
            'critical': '🔴'
        }.get(alert_level, '⚪')

        return (
            f"{color}{alert_icon} {sensor_id:<12} "
            f"{flood_icon} Reading: {reading:6.2f}m "
            f"Depth: {water_depth:5.1f}m "
            f"Status: {status:<20} "
            f"Elevation: {elevation:6.1f}m "
            f"Location: ({lat:8.4f}, {lon:9.4f}) "
            f"Time: {timestamp[11:19]}{reset_color}"
        )

    def get_sensor_statistics(self):
        """Calculate sensor network statistics"""
        if not self.last_readings:
            return {}

        stats = {
            'total_sensors': len(self.last_readings),
            'operational': 0,
            'offline': 0,
            'flooded': 0,
            'alert_levels': defaultdict(int),
            'avg_water_level': 0,
            'max_water_depth': 0,
            'min_elevation': float('inf'),
            'max_elevation': float('-inf')
        }

        water_levels = []
        water_depths = []

        for sensor_data in self.last_readings.values():
            # Count by status
            if sensor_data.get('status') == 'offline':
                stats['offline'] += 1
            else:
                stats['operational'] += 1

            # Count flooded sensors
            if sensor_data.get('is_flooded') == 'True':
                stats['flooded'] += 1

            # Count alert levels
            alert_level = sensor_data.get('alert_level', 'normal')
            stats['alert_levels'][alert_level] += 1

            # Collect readings for averages
            try:
                water_level = float(sensor_data.get('current_reading', 0))
                water_depth = float(sensor_data.get('water_depth', 0))
                elevation = float(sensor_data.get('elevation', 0))

                water_levels.append(water_level)
                water_depths.append(water_depth)

                stats['min_elevation'] = min(stats['min_elevation'], elevation)
                stats['max_elevation'] = max(stats['max_elevation'], elevation)
                stats['max_water_depth'] = max(stats['max_water_depth'], water_depth)
            except (ValueError, TypeError):
                continue

        # Calculate averages
        if water_levels:
            stats['avg_water_level'] = statistics.mean(water_levels)

        return stats

    def display_header(self):
        """Display monitoring header"""
        print("=" * 120)
        print(f"🌊 LIVE SENSOR DATA MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)

    def display_statistics(self, stats):
        """Display sensor network statistics"""
        if not stats:
            return

        print(f"\n📊 SENSOR NETWORK STATISTICS:")
        print(f"   Total Sensors: {stats['total_sensors']:<3} | "
              f"Operational: {stats['operational']:<3} | "
              f"Offline: {stats['offline']:<3} | "
              f"Flooded: {stats['flooded']:<3}")

        print(f"   Alerts - 🟢Normal: {stats['alert_levels']['normal']:<3} | "
              f"🟡Caution: {stats['alert_levels']['caution']:<3} | "
              f"🟠Warning: {stats['alert_levels']['warning']:<3} | "
              f"🔴Critical: {stats['alert_levels']['critical']:<3}")

        print(f"   Avg Water Level: {stats['avg_water_level']:.2f}m | "
              f"Max Water Depth: {stats['max_water_depth']:.1f}m | "
              f"Elevation Range: {stats['min_elevation']:.0f}m - {stats['max_elevation']:.0f}m")

    def monitor_live_data(self, show_all=False, refresh_rate=2):
        """Monitor live sensor data from Redis stream"""
        if not self.connected:
            print("❌ Not connected to Redis. Cannot monitor.")
            return

        print("🔴 STARTING LIVE SENSOR MONITORING (Ctrl+C to stop)")
        print(f"📡 Monitoring stream: {self.stream_name}")
        print(f"🔄 Refresh rate: {refresh_rate} seconds")
        print(f"📋 Display mode: {'All sensors' if show_all else 'Active sensors only'}")
        print("-" * 120)

        last_id = '$'  # Start from latest message
        self.is_monitoring = True

        try:
            while self.is_monitoring:
                try:
                    # Read new messages from stream
                    streams = self.redis_client.xread(
                        {self.stream_name: last_id},
                        block=refresh_rate * 1000,  # Convert to milliseconds
                        count=50
                    )

                    # Process new messages
                    new_data_received = False
                    for stream, messages in streams:
                        for message_id, fields in messages:
                            sensor_id = fields.get('sensor_id', 'Unknown')
                            self.last_readings[sensor_id] = fields

                            # Store in history for analysis
                            self.sensor_data_history[sensor_id].append({
                                'timestamp': datetime.now(),
                                'data': fields
                            })

                            last_id = message_id
                            new_data_received = True

                    # Always refresh display (even if no new data)
                    self.clear_screen()
                    self.display_header()

                    # Display statistics
                    stats = self.get_sensor_statistics()
                    self.display_statistics(stats)

                    print(f"\n🌊 LIVE SENSOR READINGS:")
                    print("-" * 120)

                    if self.last_readings:
                        # Sort sensors by alert level (critical first)
                        alert_priority = {'critical': 0, 'warning': 1, 'caution': 2, 'normal': 3}
                        sorted_sensors = sorted(
                            self.last_readings.items(),
                            key=lambda x: (
                                alert_priority.get(x[1].get('alert_level', 'normal'), 3),
                                x[0]  # Then by sensor ID
                            )
                        )

                        displayed_count = 0
                        for sensor_id, sensor_data in sorted_sensors:
                            # Filter display based on show_all flag
                            if not show_all:
                                # Only show active/problematic sensors
                                alert_level = sensor_data.get('alert_level', 'normal')
                                is_flooded = sensor_data.get('is_flooded', 'False') == 'True'
                                if alert_level == 'normal' and not is_flooded:
                                    continue

                            print(self.format_sensor_reading(sensor_data))
                            displayed_count += 1

                        if displayed_count == 0 and not show_all:
                            print("✅ All sensors operating normally - no alerts or flooding detected")
                            print("💡 Use 'show_all=True' to see all sensor readings")

                    else:
                        print("⏳ Waiting for sensor data...")

                    print(f"\n🔄 Last update: {datetime.now().strftime('%H:%M:%S')} | "
                          f"New data: {'Yes' if new_data_received else 'No'} | "
                          f"Press Ctrl+C to stop")

                except redis.exceptions.ResponseError as e:
                    if "NOGROUP" in str(e) or "no such key" in str(e):
                        print(f"⏳ Stream '{self.stream_name}' not found. Waiting for data...")
                        time.sleep(refresh_rate)
                    else:
                        raise e

        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")
            self.is_monitoring = False
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
            self.is_monitoring = False

    def get_sensor_history(self, sensor_id, limit=10):
        """Get historical data for a specific sensor"""
        if sensor_id not in self.sensor_data_history:
            return []

        history = list(self.sensor_data_history[sensor_id])
        return history[-limit:] if limit else history

    def export_current_data(self, filename=None):
        """Export current sensor readings to CSV"""
        if not self.last_readings:
            print("❌ No sensor data available to export")
            return

        # Prepare data for DataFrame
        export_data = []
        for sensor_id, sensor_data in self.last_readings.items():
            export_data.append({
                'sensor_id': sensor_id,
                'sensor_type': sensor_data.get('sensor_type', ''),
                'latitude': sensor_data.get('lat', ''),
                'longitude': sensor_data.get('lon', ''),
                'elevation': sensor_data.get('elevation', ''),
                'current_reading': sensor_data.get('current_reading', ''),
                'water_depth': sensor_data.get('water_depth', ''),
                'alert_level': sensor_data.get('alert_level', ''),
                'status': sensor_data.get('status', ''),
                'is_flooded': sensor_data.get('is_flooded', ''),
                'timestamp': sensor_data.get('timestamp', '')
            })

        df = pd.DataFrame(export_data)

        if filename is None:
            filename = f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        df.to_csv(filename, index=False)
        print(f"📁 Exported {len(export_data)} sensor readings to {filename}")

    def show_sensor_summary(self):
        """Show detailed sensor summary"""
        if not self.last_readings:
            print("❌ No sensor data available")
            return

        stats = self.get_sensor_statistics()

        print("\n" + "="*80)
        print("📋 DETAILED SENSOR SUMMARY")
        print("="*80)

        print(f"Network Overview:")
        print(f"  • Total Sensors: {stats['total_sensors']}")
        print(f"  • Operational: {stats['operational']} ({stats['operational']/stats['total_sensors']*100:.1f}%)")
        print(f"  • Offline: {stats['offline']} ({stats['offline']/stats['total_sensors']*100:.1f}%)")
        print(f"  • Currently Flooded: {stats['flooded']} ({stats['flooded']/stats['total_sensors']*100:.1f}%)")

        print(f"\nAlert Distribution:")
        for level, count in stats['alert_levels'].items():
            percentage = count/stats['total_sensors']*100 if stats['total_sensors'] > 0 else 0
            print(f"  • {level.title()}: {count} ({percentage:.1f}%)")

        print(f"\nEnvironmental Data:")
        print(f"  • Average Water Level: {stats['avg_water_level']:.2f}m")
        print(f"  • Maximum Water Depth: {stats['max_water_depth']:.1f}m")
        print(f"  • Elevation Range: {stats['min_elevation']:.0f}m to {stats['max_elevation']:.0f}m")

def main():
    """Main function with user interface"""
    monitor = LiveSensorMonitor()

    if not monitor.connected:
        return

    print("🌊 LIVE SENSOR DATA MONITOR")
    print("=" * 50)
    print("Available commands:")
    print("  1. Live monitoring (alerts only)")
    print("  2. Live monitoring (all sensors)")
    print("  3. Current sensor summary")
    print("  4. Export current data to CSV")
    print("  5. Exit")
    print("-" * 50)

    while True:
        try:
            choice = input("\nEnter command (1-5): ").strip()

            if choice == '1':
                monitor.monitor_live_data(show_all=False, refresh_rate=2)
            elif choice == '2':
                monitor.monitor_live_data(show_all=True, refresh_rate=2)
            elif choice == '3':
                monitor.show_sensor_summary()
            elif choice == '4':
                monitor.export_current_data()
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()