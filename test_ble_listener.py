import argparse
import asyncio
import struct
import sys
import time

try:
    from bleak import BleakClient, BleakScanner
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'bleak'. Install with: pip install bleak"
    ) from exc


SERVICE_UUID = "180A"
ACCEL_UUID = "2A58"
GYRO_UUID = "2A59"
MAG_UUID = "2A5B"


def parse_accel_or_gyro(payload: bytearray):
    if len(payload) < 12:
        return None
    # Arduino writes native floats with memcpy; Nano 33 BLE is little-endian.
    x, y, z = struct.unpack("<fff", bytes(payload[:12]))
    return x, y, z


def parse_mag(payload: bytearray):
    if len(payload) < 6:
        return None
    # Firmware packs int16 as big-endian bytes.
    x, y, z = struct.unpack(">hhh", bytes(payload[:6]))
    return x, y, z


def format_ts(start_time: float) -> str:
    dt = time.monotonic() - start_time
    return f"{dt:8.3f}s"


async def find_device(name: str, address: str, timeout: float):
    if address:
        return address

    print(f"Scanning for BLE device named '{name}' for up to {timeout:.1f}s...")
    devices = await BleakScanner.discover(timeout=timeout)
    for dev in devices:
        if dev.name == name:
            return dev.address
    return None


async def run(args):
    target = await find_device(args.name, args.address, args.scan_timeout)
    if target is None:
        raise SystemExit(
            f"Could not find BLE device '{args.name}'. "
            "Try increasing --scan-timeout or using --address."
        )

    start_time = time.monotonic()
    print(f"Connecting to {target}...")

    async with BleakClient(target, timeout=args.connect_timeout) as client:
        if not client.is_connected:
            raise SystemExit("Connection failed.")

        print("Connected.")

        if args.list_services:
            print("Services and characteristics:")
            for service in client.services:
                print(f"  Service {service.uuid}: {service.description}")
                for char in service.characteristics:
                    props = ",".join(char.properties)
                    print(f"    Char {char.uuid} ({props})")

        def on_accel(_sender, data):
            vals = parse_accel_or_gyro(data)
            if vals is None:
                print(f"[{format_ts(start_time)}] ACCEL invalid payload len={len(data)}")
                return
            x, y, z = vals
            print(f"[{format_ts(start_time)}] ACCEL x={x: .5f} y={y: .5f} z={z: .5f}")

        def on_gyro(_sender, data):
            vals = parse_accel_or_gyro(data)
            if vals is None:
                print(f"[{format_ts(start_time)}] GYRO  invalid payload len={len(data)}")
                return
            x, y, z = vals
            print(f"[{format_ts(start_time)}] GYRO  x={x: .5f} y={y: .5f} z={z: .5f}")

        def on_mag(_sender, data):
            vals = parse_mag(data)
            if vals is None:
                print(f"[{format_ts(start_time)}] MAG   invalid payload len={len(data)}")
                return
            x, y, z = vals
            print(f"[{format_ts(start_time)}] MAG   x={x:6d} y={y:6d} z={z:6d}")

        subscribed = []

        async def try_subscribe(uuid: str, callback, label: str):
            try:
                await client.start_notify(uuid, callback)
                subscribed.append(uuid)
                print(f"Subscribed: {label} ({uuid})")
                return True
            except Exception as exc:
                print(f"Warning: could not subscribe to {label} ({uuid}): {exc}")
                return False

        await try_subscribe(ACCEL_UUID, on_accel, "Accel")
        await try_subscribe(GYRO_UUID, on_gyro, "Gyro")
        await try_subscribe(MAG_UUID, on_mag, "Mag")

        if not subscribed:
            raise SystemExit("No characteristics subscribed. Check UUIDs/firmware.")

        if args.duration > 0:
            print(f"Listening for {args.duration:.1f}s...")
            await asyncio.sleep(args.duration)
        else:
            print("Listening... press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(1.0)

        for uuid in subscribed:
            try:
                await client.stop_notify(uuid)
            except Exception:
                pass

    print("Disconnected.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Connect to openMagMapper BLE sensor and print live notifications."
    )
    parser.add_argument(
        "--name",
        default="Nano33BLE_Sensor",
        help="BLE local name to scan for (default: Nano33BLE_Sensor)",
    )
    parser.add_argument(
        "--address",
        default="",
        help="BLE MAC/address to connect directly (skips scan)",
    )
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=8.0,
        help="Seconds to scan for target device",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Seconds to allow BLE connect",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Listen duration in seconds. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--list-services",
        action="store_true",
        help="Print discovered services/characteristics after connect",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
