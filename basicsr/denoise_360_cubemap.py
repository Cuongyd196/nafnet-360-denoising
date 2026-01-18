import os
import numpy as np
import torch
import imageio.v2 as imageio
import py360convert

from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import img2tensor, tensor2img


# ===============================================================
# Cubemap Utilities
# ===============================================================

def my_list2dice(faces_list):
    """
    Convert six cubemap faces into a dice (cross) layout.

    Input:
        faces_list: list of 6 faces in the order
                    [Front, Right, Back, Left, Up, Down],
                    each with shape (H, W, 3)
    Output:
        dice image with shape (3H, 4W, 3)
    """
    h, w, c = faces_list[0].shape
    dice = np.zeros((3 * h, 4 * w, c), dtype=faces_list[0].dtype)

    front, right, back, left, up, down = faces_list

    # Dice layout:
    #         [Up]
    # [Left] [Front] [Right] [Back]
    #        [Down]
    dice[0:h,       w:2*w,     :] = up
    dice[h:2*h,     0:w,       :] = left
    dice[h:2*h,     w:2*w,     :] = front
    dice[h:2*h,     2*w:3*w,   :] = right
    dice[h:2*h,     3*w:4*w,   :] = back
    dice[2*h:3*h,   w:2*w,     :] = down
    return dice


# ===============================================================
# Face-wise Denoising
# ===============================================================

def denoise_face(model, img_face):
    """
    Apply deep-learning-based denoising to a single cubemap face.

    Args:
        model: pretrained image restoration model
        img_face: cubemap face image (H, W, 3)

    Returns:
        Denoised cubemap face image (H, W, 3)
    """

    # Convert to float32
    img = img_face.astype(np.float32, copy=False)

    # Normalize to [0, 1] if needed
    if img.max() > 1.0:
        img *= (1.0 / 255.0)
    np.clip(img, 0.0, 1.0, out=img)

    # Skip nearly flat regions for numerical stability
    if img.std() < 1e-3:
        return img_face

    # Inference
    inp = img2tensor(img, bgr2rgb=False, float32=True).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model.net_g(inp)
        if not torch.isfinite(out).all():
            return img_face

    # Post-processing
    out.clamp_(0.0, 1.0)
    out_img = tensor2img(out, rgb2bgr=False, min_max=(0, 1))

    return out_img


# ===============================================================
# Main Pipeline
# ===============================================================

def main():
    # -----------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------
    opt = parse_options(is_train=False)
    opt['dist'] = False
    opt['num_gpu'] = 1

    input_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')

    # Output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    demo_dir = os.path.join(project_root, "demo")

    img_name = os.path.splitext(os.path.basename(input_path))[0]
    raw_dir = os.path.join(demo_dir, "cubemap_faces_raw", img_name)
    denoised_dir = os.path.join(demo_dir, "cubemap_faces_denoised", img_name)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(denoised_dir, exist_ok=True)

    # -----------------------------------------------------------
    # Load Model
    # -----------------------------------------------------------
    model_path = opt['path'].get('pretrain_network_g')
    if not os.path.exists(model_path):
        print(f"Missing pretrained model: {model_path}")
        return

    print("Loading NAFNet model...")
    model = create_model(opt)
    model.net_g.eval()

    # -----------------------------------------------------------
    # Load ERP Image
    # -----------------------------------------------------------
    print(f"Reading input image: {input_path}")
    erp_img = imageio.imread(input_path)
    h_erp, w_erp, _ = erp_img.shape

    # -----------------------------------------------------------
    # ERP → Cubemap
    # -----------------------------------------------------------
    face_w = 512
    print(f"Converting ERP to cubemap (face size = {face_w})")

    faces_list = py360convert.e2c(
        erp_img,
        face_w=face_w,
        mode='bilinear',
        cube_format='list'
    )

    face_names = ['Front', 'Right', 'Back', 'Left', 'Up', 'Down']

    # Save raw cubemap faces
    for i, face in enumerate(faces_list):
        face_u8 = face
        if face_u8.dtype != np.uint8:
            face_u8 = np.clip(face_u8, 0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(raw_dir, f"{face_names[i]}.png"), face_u8)

    # -----------------------------------------------------------
    # Face-wise Denoising
    # -----------------------------------------------------------
    print("Denoising cubemap faces...")
    denoised_faces = []
    for i, face in enumerate(faces_list):
        print(f"  - {face_names[i]}")
        denoised_faces.append(denoise_face(model, face))

    # Save denoised cubemap faces
    for i, face in enumerate(denoised_faces):
        face_u8 = np.clip(face, 0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(denoised_dir, f"{face_names[i]}.png"), face_u8)

    # -----------------------------------------------------------
    # Cubemap → ERP Reconstruction
    # -----------------------------------------------------------
    dice_img = my_list2dice(denoised_faces)
    print("Reconstructing ERP image...")
    erp_denoised = py360convert.c2e(
        dice_img,
        h=h_erp,
        w=w_erp,
        mode='bilinear'
    )

    # -----------------------------------------------------------
    # Blending for Detail Preservation
    # -----------------------------------------------------------
    orig = erp_img.astype(np.float32)
    clean = erp_denoised.astype(np.float32)
    alpha = 0.3
    blended = np.clip(orig * alpha + clean * (1 - alpha), 0, 255).astype(np.uint8)

    # -----------------------------------------------------------
    # Save Results
    # -----------------------------------------------------------
    imageio.imwrite(output_path.replace(".jpg", "_erp_denoised.jpg"),
                    erp_denoised, quality=90, subsampling=0)
    imageio.imwrite(output_path.replace(".jpg", f"_blended_{alpha}.jpg"),
                    blended, quality=90, subsampling=0)

    print("Processing completed successfully.")


if __name__ == '__main__':
    main()
