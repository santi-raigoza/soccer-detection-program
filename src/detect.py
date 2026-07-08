import argparse
from utils.pipeline import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and track soccer ball in a video.")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to write annotated output video")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLOv8 weights (default: models/best.pt)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--trail-len", type=int, default=15, help="Trajectory trail length in frames (default: 15)")
    args = parser.parse_args()

    print(f"Processing: {args.input}")
    print(f"Model:      {args.model}  conf={args.conf}  trail={args.trail_len}")
    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        trail_len=args.trail_len,
    )
    print(f"Saved to:   {args.output}")


if __name__ == "__main__":
    main()
