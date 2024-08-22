import argparse
import logging
import os
import random

from diffusionkit.mlx import FluxPipeline

logging.getLogger("diffusionkit.mlx").setLevel(logging.WARNING)
logging.getLogger("diffusionkit.mlx.mmdit").setLevel(logging.WARNING)


HEIGHT = 768
WIDTH = 1360
CFG_WEIGHT = 0.0  # for FLUX.1-schnell, 5. for SD3


def generate_from_prompt(pipeline: FluxPipeline, prompt, steps, seed, output_path):
    image, _ = pipeline.generate_image(
        prompt,
        cfg_weight=CFG_WEIGHT,
        num_steps=steps,
        seed=seed,
        latent_size=(HEIGHT // 8, WIDTH // 8),
        verbose=False,
    )
    image.save(output_path)


def generate_from_image(
    pipeline: FluxPipeline, image_path, prompt, steps, seed, denoise, output_path
):
    image, _ = pipeline.generate_image(
        prompt,
        cfg_weight=CFG_WEIGHT,
        num_steps=steps,
        seed=seed,
        latent_size=(HEIGHT // 8, WIDTH // 8),
        verbose=False,
        image_path=image_path,
        denoise=denoise,
    )
    image.save(output_path)


def image_loop(
    pipeline, initial_image_path, output_dir, loop_size, steps=10, denoise=0.7
):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i in range(1, loop_size):
        im_path = f"{output_dir}/{i}.png"
        seed = random.randint(1, 99999)
        if i == 1:
            generate_from_image(
                pipeline,
                initial_image_path,
                "",
                steps,
                seed,
                denoise,
                im_path,
            )
        else:
            prev_image_path = f"{output_dir}/{i-1}.png"
            generate_from_image(
                pipeline,
                prev_image_path,
                "",
                steps,
                seed,
                denoise,
                im_path,
            )


def parse_arguments():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="mode", help="Choose mode: 'prompt' or 'image_loop'"
    )

    # prompt mode
    prompt_parser = subparsers.add_parser("prompt", help="Generate image from a prompt")
    prompt_parser.add_argument(
        "prompt", type=str, help="The text prompt for image generation"
    )
    prompt_parser.add_argument(
        "output_path", type=str, help="Path to save the generated image"
    )
    prompt_parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps for generation (default: 10)",
    )
    prompt_parser.add_argument(
        "--seed", type=int, default=None, help="Seed for generation (default: random)"
    )

    # image_loop mode
    loop_parser = subparsers.add_parser(
        "image_loop", help="Generate images in a loop starting from an initial image"
    )
    loop_parser.add_argument(
        "initial_image_path", type=str, help="Path to the initial image"
    )
    loop_parser.add_argument(
        "output_dir", type=str, help="Directory to save the looped images"
    )
    loop_parser.add_argument(
        "loop_size", type=int, help="Number of images to generate in the loop"
    )
    loop_parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps for each generation (default: 10)",
    )
    loop_parser.add_argument(
        "--denoise", type=float, default=0.7, help="Denoising strength (default: 0.7)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    pipeline = FluxPipeline(
        model="argmaxinc/stable-diffusion",
        shift=1.0,
        model_version="FLUX.1-schnell",
        low_memory_mode=True,
        a16=True,
        w16=True,
    )

    if args.mode == "prompt":
        seed = args.seed if args.seed is not None else random.randint(1, 99999)
        generate_from_prompt(pipeline, args.prompt, args.steps, seed, args.output_path)
    elif args.mode == "image_loop":
        image_loop(
            pipeline,
            args.initial_image_path,
            args.output_dir,
            args.loop_size,
            args.steps,
            args.denoise,
        )
    else:
        print("Invalid mode selected. Use 'prompt' or 'image_loop'.")
