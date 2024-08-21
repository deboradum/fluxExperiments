import os
import random

from diffusionkit.mlx import FluxPipeline

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


def image_loop(initial_image_path, output_dir, loop_size, steps=10, denoise=0.7):
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


if __name__ == "__main__":
    pipeline = FluxPipeline(
        model="argmaxinc/stable-diffusion",
        shift=1.0,
        model_version="FLUX.1-schnell",
        low_memory_mode=True,
        a16=True,
        w16=True,
    )

    initial_image_path = "images/prompt_6_steps_5_seed_54448.png"
    output_dir = "cyberpunk"
    image_loop(initial_image_path, output_dir, 500)
