import asyncio
import struct
import sys
import threading
import time

import numpy as np

try:
    from bleak import BleakClient, BleakScanner
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'bleak'. Install with: pip install bleak"
    ) from exc


SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
MAGMLX_UUID = "19B10004-E8F2-537E-4F6C-D104768A1214"


def normalize_ble_name(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def parse_magmlx_payload(payload):
    if len(payload) < 6:
        return None
    x, y, z = struct.unpack(">hhh", bytes(payload[:6]))
    return np.array([x, y, z], dtype=float)


async def find_device(name: str, address: str, timeout: float):
    if address:
        dev = await BleakScanner.find_device_by_address(address, timeout=timeout)
        if dev is not None:
            advertised = dev.metadata.get("uuids", []) if hasattr(dev, "metadata") and isinstance(dev.metadata, dict) else []
            return dev, advertised
        return address, None

    devices = await BleakScanner.discover(timeout=timeout)
    target_name = normalize_ble_name(name)
    for dev in devices:
        dev_name = normalize_ble_name(dev.name or "")
        if dev_name and (dev_name == target_name or dev_name in target_name or target_name in dev_name):
            advertised = dev.metadata.get("uuids", []) if hasattr(dev, "metadata") and isinstance(dev.metadata, dict) else []
            return dev, advertised
    return None, []


class BleMagmlxSource:
    def __init__(
        self,
        name: str = "Nano33BLE_Sensor",
        address: str = "",
        scan_timeout: float = 20.0,
        connect_timeout: float = 10.0,
        connect_retries: int = 3,
        retry_delay: float = 2.0,
        poll_interval: float = 0.2,
    ):
        self.name = name
        self.address = address
        self.scan_timeout = scan_timeout
        self.connect_timeout = connect_timeout
        self.connect_retries = max(1, int(connect_retries))
        self.retry_delay = float(retry_delay)
        self.poll_interval = max(0.05, float(poll_interval))

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._thread = None
        self._latest_vector = None
        self._latest_update_time = None
        self._sequence = 0
        self._last_error = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, name="ble-magmlx", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def last_error(self):
        return self._last_error

    @property
    def is_connected(self):
        return self._connected_event.is_set()

    def snapshot(self):
        with self._lock:
            if self._latest_vector is None:
                return None
            return self._latest_vector.copy(), self._latest_update_time, self._sequence

    def _store_sample(self, payload):
        vector = parse_magmlx_payload(payload)
        if vector is None:
            return
        with self._lock:
            self._latest_vector = vector
            self._latest_update_time = time.monotonic()
            self._sequence += 1

    def _thread_main(self):
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._last_error = exc
        finally:
            self._connected_event.clear()

    async def _connect(self):
        target, _advertised_uuids = await find_device(self.name, self.address, self.scan_timeout)
        if target is None:
            raise RuntimeError(f"Could not find BLE device '{self.name}'.")

        if sys.platform.startswith("win"):
            connect_modes = [
                {"label": "cached-services", "winrt": {"use_cached_services": True}},
                {"label": "fresh-services", "winrt": {"use_cached_services": False}},
            ]
        else:
            connect_modes = [{"label": "default", "winrt": None}]

        connect_errors = []
        for attempt in range(1, self.connect_retries + 1):
            attempt_target = target
            if isinstance(target, str):
                refreshed = await BleakScanner.find_device_by_address(target, timeout=max(3.0, self.scan_timeout / 2.0))
                if refreshed is not None:
                    attempt_target = refreshed

            for mode in connect_modes:
                try:
                    client_kwargs = {
                        "timeout": self.connect_timeout,
                    }
                    if mode["winrt"] is not None:
                        client_kwargs["winrt"] = mode["winrt"]
                    client = BleakClient(attempt_target, **client_kwargs)
                    await client.connect()
                    if client.is_connected:
                        return client
                except Exception as exc:
                    connect_errors.append((f"attempt {attempt} {mode['label']}", repr(exc)))

            if attempt < self.connect_retries:
                await asyncio.sleep(max(0.1, self.retry_delay))

        error_lines = ["All BLE connection attempts failed:"]
        for label, err in connect_errors:
            error_lines.append(f"  {label}: {err}")
        raise RuntimeError("\n".join(error_lines))

    async def _run_connected_client(self, client):
        self._connected_event.set()

        def on_magmlx(_sender, data):
            self._store_sample(data)

        subscribed = False
        try:
            await client.start_notify(MAGMLX_UUID, on_magmlx)
            subscribed = True
        except Exception:
            subscribed = False

        if subscribed:
            while not self._stop_event.is_set() and client.is_connected:
                await asyncio.sleep(0.1)
            return

        while not self._stop_event.is_set() and client.is_connected:
            try:
                data = await client.read_gatt_char(MAGMLX_UUID)
                self._store_sample(data)
            except Exception:
                await asyncio.sleep(self.poll_interval)
                continue
            await asyncio.sleep(self.poll_interval)

    async def _run(self):
        while not self._stop_event.is_set():
            client = None
            try:
                client = await self._connect()
                self._last_error = None
                await self._run_connected_client(client)
            except Exception as exc:
                self._last_error = exc
            finally:
                self._connected_event.clear()
                if client is not None:
                    try:
                        if client.is_connected:
                            await client.disconnect()
                    except Exception:
                        pass

            if not self._stop_event.is_set():
                await asyncio.sleep(max(0.2, self.retry_delay))
