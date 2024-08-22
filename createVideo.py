import argparse
import os
import re

from moviepy.editor import ImageSequenceClip

def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def create_video_from_images(image_paths, output_path, fps):
    clip = ImageSequenceClip(image_paths, fps=fps)
    clip.write_videofile(output_path, codec="libx264")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", type=str, help="Directory containing the PNG images."
    )
    parser.add_argument("fps", type=int, help="Frames per second for the video.")
    parser.add_argument("output", type=str, help="Output video file name.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    files = [
        os.path.join(args.directory, f)
        for f in os.listdir(args.directory)
        if f.endswith(".png")
    ]
    image_paths = sorted(files, key=extract_number)

    if not image_paths:
        print("No PNG files found in the directory.")
        exit()

    create_video_from_images(image_paths, args.output, args.fps)
