# Quick script to capture frames using the same OpenCV setup as runExperiment.py, for camera calibration purposes.

import argparse
import os
import time
from datetime import datetime

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture frames using the same OpenCV setup as runExperiment.py"
    )
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height (default: 720)")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("data", "calibration_capture"),
        help="Directory to save captured frames",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {args.camera}")
        return 1

    # Match runExperiment.py camera property setup.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("Camera opened.")
    print(f"Requested: {args.width}x{args.height} @ camera index {args.camera}")
    print(f"Actual:    {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
    print(f"Saving frames to: {os.path.abspath(args.outdir)}")
    print("Controls: [space]=save frame, [q]=quit")

    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame from camera.")
            break

        display = frame.copy()
        overlay_text = (
            f"req {args.width}x{args.height} | actual {actual_width}x{actual_height} | "
            f"fps {actual_fps:.2f} | saved {saved_count}"
        )
        cv2.putText(display, overlay_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Capture (match runExperiment)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = os.path.join(args.outdir, f"frame_{timestamp}.png")
            ok = cv2.imwrite(out_path, frame)
            if ok:
                saved_count += 1
                print(f"Saved: {out_path}")
            else:
                print(f"ERROR: Failed to save {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved_count} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
