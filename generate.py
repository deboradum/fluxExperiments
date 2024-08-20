import random

from diffusionkit.mlx import FluxPipeline
from time import perf_counter

HEIGHT = 768
WIDTH = 1360
CFG_WEIGHT = 0.0  # for FLUX.1-schnell, 5. for SD3
PROMPTS = [
    "A futuristic, dystopian cityscape reminiscent of 'Ghost in the Shell' and the neon-lit streets of Tokyo. The image showcases winding, shadowy alleyways bathed in the faint glow of neon signs, reflecting off rain-soaked pavement. Towering skyscrapers dominate the skyline, their surfaces alive with animated billboards. The scene feels tense and mysterious, with a few indistinct figures shrouded in mist, navigating the gritty, urban labyrinth.",
    "A cyberpunk-inspired cityscape set in a dystopian future, echoing the ambiance of 'Ghost in the Shell' and the crowded streets of Hong Kong. Narrow, maze-like alleys are dimly lit by flickering neon signs, casting colorful reflections on the wet concrete. The towering buildings are covered in digital advertisements, adding to the overwhelming sense of urban decay. The atmosphere is tense, with shadowy silhouettes moving through the foggy streets, evoking a world on the edge of collapse.",
    "A dark, hyper-modern cityscape with influences from 'Ghost in the Shell' and the iconic urban sprawl of Hong Kong. The scene features tight, labyrinthine alleyways, their walls lined with flickering neon lights. The wet streets below reflect the dim glow, creating an eerie, cinematic atmosphere. Towering skyscrapers loom above, covered in holographic advertisements. The streets are nearly empty, save for a few shadowy figures moving through the mist, adding a sense of unease and mystery.",
    "A dystopian urban landscape inspired by 'Ghost in the Shell' and the dense streets of Tokyo. The scene is set in narrow, dark alleyways, where neon lights flicker and cast a dim, ghostly glow on the rain-soaked streets. Towering skyscrapers with pulsating advertisements rise in the background, creating a sense of claustrophobia and tension. The few figures present are obscured by shadows and mist, heightening the sense of futuristic decay and gritty realism.",
    "A gritty, cyberpunk cityscape inspired by 'Ghost in the Shell' and the neon-lit streets of Hong Kong. The scene captures a network of narrow alleyways, their walls lined with glowing neon signs that cast an eerie light on the wet pavement. Massive skyscrapers loom in the background, adorned with digital ads that flicker in and out. The streets are sparsely populated, with shadowy figures barely visible through the mist, adding to the atmosphere of mystery and tension.",
    "A futuristic dystopian cityscape with influences from 'Ghost in the Shell' and the crowded, neon-lit streets of Tokyo. The image features narrow, winding alleys bathed in the soft glow of neon lights, with rain-soaked streets reflecting the dim light. Skyscrapers tower in the distance, their surfaces alive with flickering advertisements. The atmosphere is tense and mysterious, with shadowy figures moving through the fog, evoking a world steeped in gritty realism and futuristic decay.",
    "A hyper-modern, dystopian urban landscape inspired by 'Ghost in the Shell' and the vibrant streets of Hong Kong. The scene is set in tight, shadowy alleyways, illuminated by flickering neon lights that reflect off the wet ground. Towering skyscrapers with holographic billboards loom ominously in the background. The streets are nearly empty, with only a few indistinct figures moving through the mist, creating a cinematic atmosphere of tension and mystery.",
    "A dark, dystopian cityscape with elements from 'Ghost in the Shell' and the bustling streets of Tokyo. The image features a maze of narrow alleys, dimly lit by neon signs that cast an eerie glow on the rain-slicked streets. Towering skyscrapers in the distance are covered in flickering advertisements, adding to the sense of urban decay. The streets are sparsely populated, with shadowy silhouettes moving through the fog, evoking a sense of gritty realism and foreboding.",
    "A cyberpunk-inspired dystopian cityscape reminiscent of 'Ghost in the Shell' and the neon-soaked streets of Hong Kong. The scene showcases narrow, labyrinthine alleys bathed in the dim light of neon signs, reflecting off wet pavement. Towering skyscrapers loom in the background, their surfaces covered in animated advertisements. The atmosphere is tense and mysterious, with a few shadowy figures navigating the misty streets, creating a sense of gritty, futuristic decay.",
    "A futuristic, hyper-modern cityscape with influences from 'Ghost in the Shell' and the crowded streets of Tokyo. The image captures narrow, dark alleyways, illuminated by flickering neon lights that cast a ghostly glow on the rain-soaked streets below. Massive skyscrapers with pulsating advertisements dominate the background, adding to the sense of urban decay. The few figures present are obscured by shadows and mist, heightening the sense of tension and mystery.",
]


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


def generate_from_image(pipeline: FluxPipeline, image_path, prompt, steps, seed, output_path):
    image, _ = pipeline.generate_image(
        prompt,
        cfg_weight=CFG_WEIGHT,
        num_steps=steps,
        seed=seed,
        latent_size=(HEIGHT // 8, WIDTH // 8),
        verbose=False,
        image_path=image_path,
    )
    image.save(output_path)


if __name__ == "__main__":
    pipeline = FluxPipeline(
        model="argmaxinc/stable-diffusion",
        shift=1.0,
        model_version="FLUX.1-schnell",
        low_memory_mode=True,
        a16=True,
        w16=True,
    )

    for i, prompt in enumerate(PROMPTS):
        s = random.randint(1, 99999)
        for steps in range(2, 6):
            output_path = f"images/prompt_{i}_steps_{steps}_seed_{s}.png"
            generate_from_prompt(pipeline, prompt, steps, s, output_path, verbose=True)
