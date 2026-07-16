import argparse
import sys
import time

import serial
import ommFuncs as omm


def build_parser():
    parser = argparse.ArgumentParser(
        description="Live serial stream monitor for openMagMapper debugging."
    )
    parser.add_argument(
        "--port",
        default="",
        help="Serial port name (for example COM11). If omitted, auto-detect is used.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baud rate (default: 115200).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.2,
        help="Serial read timeout in seconds.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw lines only (skip packet parsing).",
    )
    return parser


def resolve_port(port_arg: str) -> str:
    if port_arg:
        return port_arg

    print("Auto-detecting serial port...")
    detected = ""
    try:
        from serial.tools import list_ports

        ports = list(list_ports.comports(include_links=False))
        for port in ports:
            print(f"Found port: {port.device}")
        if ports:
            detected = ports[-1].device
    except Exception:
        detected = ""

    if not detected:
        # Fallback to project helper if serial.tools is unavailable in this environment.
        detected = omm.getSerialPort()

    if not detected:
        raise SystemExit("No serial port found. Provide --port explicitly.")
    print(f"Using detected port: {detected}")
    return detected


def monitor_stream(port: str, baud: int, timeout: float, raw_only: bool):
    print(f"Opening {port} at {baud} baud")
    ser = serial.Serial(port, baud, timeout=timeout)
    time.sleep(0.1)
    ser.reset_input_buffer()

    print("Streaming... press Ctrl+C to stop.")
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        if raw_only:
            print(line)
            continue

        values, parsed = omm.parse_serial_packet(line)
        if values is None:
            print(f"BAD  {line}")
            continue

        parts = ", ".join(f"{v:.3f}" for v in values[:4])
        print(f"OK   {parsed} | [{parts}] len={len(values)}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        port = resolve_port(args.port)
        monitor_stream(port, args.baud, args.timeout, args.raw)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
