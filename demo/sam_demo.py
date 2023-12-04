import json

import cv2
import gradio as gr
import hydra
import numpy as np
import yaml
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from SAMTA.ext.sam import SAM_predictor


def plot_mask(img, masks, colors=None, alpha=0.5) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        color for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    if colors is None:
        colors = np.random.random((masks.shape[0], 3)) * 255
    else:
        if colors.shape[0] < masks.shape[0]:
            raise RuntimeError(
                f"colors count: {colors.shape[0]} is less than masks count: {masks.shape[0]}"
            )
    for mask, color in zip(masks, colors):
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)

    return img.astype(np.uint8)


def show_anns(anns, image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    image = np.concatenate(
        (
            image,
            np.zeros(
                (image.shape[0], image.shape[1], 1),
            ),
        ),
        axis=2,
    )
    masked_img = plot_mask(image, sorted_anns)

    return masked_img


def get_config(cfg: DictConfig):
    yaml_str = OmegaConf.to_yaml(cfg)
    yaml_dict = yaml.safe_load(yaml_str)
    json_str = json.dumps(yaml_dict, sort_keys=False, indent=4, separators=(",", ": "))

    print(type(json_str))
    print(json_str)
    print("---------------------")

    return json_str


def demo_with_sam(image: gr.Image, config: gr.JSON):
    predictor = SAM_predictor.SAM_predictor(config["sam"])

    masks = predictor.generator_masks(image)

    img = show_anns(masks, image)

    return img


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 格式转换为json
    json_str = get_config(cfg)

    sam_demo = gr.Interface(
        fn=demo_with_sam,
        inputs=[
            gr.Image(),
            gr.JSON(json_str, visible=False),
        ],
        outputs="image",
        title="SAM-TA: Tracking Anything with SAM",
    )

    sam_demo.launch(server_name="0.0.0.0", server_port=12701, share=True)


if __name__ == "__main__":
    main()
