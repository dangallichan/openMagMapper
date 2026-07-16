import argparse
import asyncio
import struct
import sys
import time
import traceback

try:
    from bleak import BleakClient, BleakScanner
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'bleak'. Install with: pip install bleak"
    ) from exc


SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
ACCEL_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
GYRO_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"
MAG_UUID = "19B10003-E8F2-537E-4F6C-D104768A1214"
MAGMLX_UUID = "19B10004-E8F2-537E-4F6C-D104768A1214"


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


def normalize_ble_name(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


async def find_device(name: str, address: str, timeout: float):
    if address:
        dev = await BleakScanner.find_device_by_address(address, timeout=timeout)
        if dev is not None:
            advertised = dev.metadata.get("uuids", []) if hasattr(dev, "metadata") and isinstance(dev.metadata, dict) else []
            return dev, advertised
        # Fallback to raw address if lookup fails.
        return address, None

    print(f"Scanning for BLE device named '{name}' for up to {timeout:.1f}s...")
    devices = await BleakScanner.discover(timeout=timeout)
    target_name = normalize_ble_name(name)
    for dev in devices:
        dev_name = normalize_ble_name(dev.name or "")
        if dev_name and (dev_name == target_name or dev_name in target_name or target_name in dev_name):
            advertised = dev.metadata.get("uuids", []) if hasattr(dev, "metadata") and isinstance(dev.metadata, dict) else []
            if advertised:
                print("Scan match advertised service UUIDs:")
                for uuid in advertised:
                    print(f"  {uuid}")
            else:
                print("Scan match found, but no advertised service UUIDs were reported.")
            return dev, advertised
    return None, []


async def run(args):
    target, advertised_uuids = await find_device(args.name, args.address, args.scan_timeout)
    if target is None:
        raise SystemExit(
            f"Could not find BLE device '{args.name}'. "
            "Try increasing --scan-timeout or using --address."
        )

    target_label = getattr(target, "address", str(target))
    start_time = time.monotonic()
    print(f"Connecting to {target_label}...")
    disconnected_event = asyncio.Event()

    def on_disconnect(_client):
        disconnected_event.set()
        print(f"[{format_ts(start_time)}] Disconnected from device.")

    async def connect_once(connect_target, winrt_opts=None):
        client_kwargs = {
            "timeout": args.connect_timeout,
            "disconnected_callback": on_disconnect,
        }
        if winrt_opts is not None:
            client_kwargs["winrt"] = winrt_opts
        client = BleakClient(connect_target, **client_kwargs)
        await client.connect()
        return client

    client = None
    connect_errors = []
    if sys.platform.startswith("win"):
        connect_modes = [
            {"label": "cached-services", "winrt": {"use_cached_services": True}},
            {"label": "fresh-services", "winrt": {"use_cached_services": False}},
        ]
    else:
        connect_modes = [{"label": "default", "winrt": None}]

    for attempt in range(1, max(1, args.connect_retries) + 1):
        # Refresh BLEDevice handle each attempt when we only have an address string.
        attempt_target = target
        if isinstance(target, str):
            refreshed = await BleakScanner.find_device_by_address(target, timeout=max(3.0, args.scan_timeout / 2.0))
            if refreshed is not None:
                attempt_target = refreshed

        for mode in connect_modes:
            try:
                print(f"Connect attempt {attempt}/{max(1, args.connect_retries)} mode: {mode['label']}")
                client = await connect_once(attempt_target, winrt_opts=mode["winrt"])
                break
            except Exception as exc:
                connect_errors.append((f"attempt {attempt} {mode['label']}", repr(exc)))

        if client is not None:
            break

        if attempt < max(1, args.connect_retries):
            await asyncio.sleep(max(0.1, args.retry_delay))

    if client is None:
        print("All connection attempts failed:")
        for label, err in connect_errors:
            print(f"  {label}: {err}")
        raise SystemExit("Unable to connect to BLE device.")

    try:
        if not client.is_connected:
            raise SystemExit("Connection failed.")

        print("Connected.")
        print(f"Connected address: {client.address}")
        if advertised_uuids:
            print("Advertised UUIDs seen during scan:")
            for uuid in advertised_uuids:
                print(f"  {uuid}")

        print("Discovered characteristics:")
        discovered_chars = []
        for service in client.services:
            for char in service.characteristics:
                props = ",".join(char.properties)
                discovered_chars.append((str(service.uuid), str(char.uuid), props))
                print(f"  Service {service.uuid} | Char {char.uuid} ({props})")

        expected_uuids = [ACCEL_UUID, GYRO_UUID, MAG_UUID, MAGMLX_UUID]
        discovered_uuid_set = {char_uuid.lower() for _, char_uuid, _ in discovered_chars}
        print("Expected characteristic presence:")
        for uuid in expected_uuids:
            state = "FOUND" if uuid.lower() in discovered_uuid_set else "MISSING"
            print(f"  {uuid}: {state}")

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
            vals = parse_accel_or_gyro(data)
            if vals is None:
                print(f"[{format_ts(start_time)}] MAG    invalid payload len={len(data)}")
                return
            x, y, z = vals
            print(f"[{format_ts(start_time)}] MAG    x={x: .5f} y={y: .5f} z={z: .5f}")

        def on_magmlx(_sender, data):
            vals = parse_mag(data)
            if vals is None:
                print(f"[{format_ts(start_time)}] MAGMLX invalid payload len={len(data)}")
                return
            x, y, z = vals
            print(f"[{format_ts(start_time)}] MAGMLX x={x:6d} y={y:6d} z={z:6d}")

        subscribed = []
        pollers = []

        def get_characteristic(uuid: str):
            try:
                return client.services.get_characteristic(uuid)
            except Exception:
                return None

        def can_read(uuid: str) -> bool:
            ch = get_characteristic(uuid)
            if ch is None:
                return False
            props = {p.lower() for p in ch.properties}
            return "read" in props

        async def polling_loop(uuid: str, label: str, parser, formatter):
            while not disconnected_event.is_set() and client.is_connected:
                try:
                    data = await client.read_gatt_char(uuid)
                    vals = parser(data)
                    if vals is None:
                        print(f"[{format_ts(start_time)}] {label} invalid payload len={len(data)}")
                    else:
                        print(formatter(vals))
                except Exception as exc:
                    msg = str(exc)
                    if "Not connected" in msg or not client.is_connected:
                        disconnected_event.set()
                        break
                    print(f"[{format_ts(start_time)}] {label} read error: {exc}")
                await asyncio.sleep(max(args.poll_interval, 0.05))

        async def try_subscribe(uuid: str, callback, label: str):
            try:
                await client.start_notify(uuid, callback)
                subscribed.append(uuid)
                print(f"Subscribed: {label} ({uuid})")
                return True
            except Exception as exc:
                print(f"Warning: could not subscribe to {label} ({uuid}): {exc}")
                if can_read(uuid):
                    print(f"Falling back to polling reads for {label} ({uuid}).")
                    return False
                return False

        accel_sub = await try_subscribe(ACCEL_UUID, on_accel, "Accel")
        gyro_sub = await try_subscribe(GYRO_UUID, on_gyro, "Gyro")
        mag_sub = await try_subscribe(MAG_UUID, on_mag, "MAG")
        magmlx_sub = await try_subscribe(MAGMLX_UUID, on_magmlx, "MAGMLX")

        if not accel_sub and can_read(ACCEL_UUID):
            pollers.append(
                asyncio.create_task(
                    polling_loop(
                        ACCEL_UUID,
                        "ACCEL",
                        parse_accel_or_gyro,
                        lambda vals: f"[{format_ts(start_time)}] ACCEL x={vals[0]: .5f} y={vals[1]: .5f} z={vals[2]: .5f}",
                    )
                )
            )

        if not gyro_sub and can_read(GYRO_UUID):
            pollers.append(
                asyncio.create_task(
                    polling_loop(
                        GYRO_UUID,
                        "GYRO ",
                        parse_accel_or_gyro,
                        lambda vals: f"[{format_ts(start_time)}] GYRO  x={vals[0]: .5f} y={vals[1]: .5f} z={vals[2]: .5f}",
                    )
                )
            )

        if not mag_sub and can_read(MAG_UUID):
            pollers.append(
                asyncio.create_task(
                    polling_loop(
                        MAG_UUID,
                        "MAG   ",
                        parse_accel_or_gyro,
                        lambda vals: f"[{format_ts(start_time)}] MAG    x={vals[0]: .5f} y={vals[1]: .5f} z={vals[2]: .5f}",
                    )
                )
            )

        if not magmlx_sub and can_read(MAGMLX_UUID):
            pollers.append(
                asyncio.create_task(
                    polling_loop(
                        MAGMLX_UUID,
                        "MAGMLX",
                        parse_mag,
                        lambda vals: f"[{format_ts(start_time)}] MAGMLX x={vals[0]:6d} y={vals[1]:6d} z={vals[2]:6d}",
                    )
                )
            )

        if not subscribed and not pollers:
            raise SystemExit("No characteristics subscribed or readable. Check UUIDs/firmware.")

        if args.duration > 0:
            print(f"Listening for {args.duration:.1f}s...")
            try:
                await asyncio.wait_for(disconnected_event.wait(), timeout=args.duration)
            except asyncio.TimeoutError:
                pass
        else:
            print("Listening... press Ctrl+C to stop.")
            while not disconnected_event.is_set() and client.is_connected:
                await asyncio.sleep(1.0)

        for uuid in subscribed:
            try:
                await client.stop_notify(uuid)
            except Exception:
                pass

        for task in pollers:
            task.cancel()
        if pollers:
            await asyncio.gather(*pollers, return_exceptions=True)

    finally:
        if client is not None and client.is_connected:
            try:
                await client.disconnect()
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
        # default="42:D5:F4:FB:16:1C",
        default="D1:A3:04:CC:25:EC",
        # default="",
        help="BLE MAC/address to connect directly (skips scan)",
    )
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=25.0,
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
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="Polling interval in seconds when notify is unavailable",
    )
    parser.add_argument(
        "--connect-retries",
        type=int,
        default=3,
        help="Number of connect retry attempts before failing",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Delay between connect retry attempts in seconds",
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
        print(f"Error: {type(exc).__name__}: {exc!r}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
