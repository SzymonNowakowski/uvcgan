import os
import cv2
import torch
import numpy as np
from PIL import Image

from uvcgan.utils.model_state import ModelState
from uvcgan.eval.funcs import start_model_eval
from bio_PLB.data.bio_dataset import GlobalAndInstanceNorm


def unnormalize_to_hwc(tensor, mean, std):
    img = tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img.squeeze(0)

    # Odwrócenie normalizacji globalnej
    img = (img * std) + mean

    # Transpozycja do formatu HWC (wysokość, szerokość, kanały)
    if img.ndim == 3:
        img = img.transpose((1, 2, 0))
    return img


def run_bio_inference():
    # --- 1. KONFIGURACJA ---
    model_path = 'outdir/synthetic2real/model_d(unpaired-bio)_m(cyclegan)_d(basic)_g(vit-unet)_unpaired-bio_vit-unet-12-self-lsgan-paper-cycle_high-160px'
    input_dir = 'input/images'
    output_dir = 'outputdir/images'
    os.makedirs(output_dir, exist_ok=True)

    # Statystyki domen
    SYNTH_STATS = {'mean': 0.7367, 'std': 0.1922}
    REAL_STATS = {'mean': 0.2363, 'std': 0.1224}

    # Mapowanie kluczy na odpowiednie statystyki dla poprawnego unnorm
    targets = {
        'real_a': SYNTH_STATS,
        'fake_b': REAL_STATS,
        'reco_a': SYNTH_STATS,
        'real_b': REAL_STATS,
        'fake_a': SYNTH_STATS,
        'reco_b': REAL_STATS
    }

    model_state = ModelState.from_str('eval')
    args, model, _ = start_model_eval(model_path, epoch=-1, model_state=model_state)
    model.eval()

    bio_norm = GlobalAndInstanceNorm(SYNTH_STATS['mean'], SYNTH_STATS['std'])

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in image_files:
        name_base, ext = os.path.splitext(fname)

        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        input_tensor = torch.from_numpy(img).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        input_tensor = bio_norm(input_tensor).to(model.device)

        with torch.no_grad():
            # Podwójne wejście zamiast dummy
            model.set_input([input_tensor, input_tensor])
            model.forward_nograd()

            # Iteracja po liście obrazów z dopisywaniem suffixu
            for key, stats in targets.items():
                tensor = getattr(model.images, key, None)

                if tensor is not None:
                    # Unnorm z dopasowanymi statystykami
                    output_hwc = unnormalize_to_hwc(tensor[0], stats['mean'], stats['std'])

                    # Skalowanie do uint8
                    output_final = np.clip(output_hwc * 255, 0, 255).astype('uint8')

                    # FLATTEN: Spłaszczenie do 1 kanału (H, W)
                    gray_out = np.squeeze(output_final)

                    # Budowa nazwy pliku: oryginał_klucz.png
                    new_fname = f"{name_base}_{key}{ext}"
                    save_path = os.path.join(output_dir, new_fname)

                    # Zapis w trybie 'L' (Grayscale)
                    res_img = Image.fromarray(gray_out, mode='L')
                    res_img.save(save_path)

        print(f"Zapisano wszystkie wersje dla: {fname}")


if __name__ == "__main__":
    run_bio_inference()