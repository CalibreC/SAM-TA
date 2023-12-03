import hydra
import numpy as np
import yaml
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
import gradio as gr
import json
from SAMTA.ext.sam import SAM_predictor


def show_anns(anns, image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    print(type(img))
    print(img.shape)

    return img


def get_config(cfg: DictConfig):
    yaml_str = OmegaConf.to_yaml(cfg)
    yaml_dict = yaml.safe_load(yaml_str)
    json_str = json.dumps(yaml_dict, sort_keys=False, indent=4, separators=(',', ': '))

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
        # examples=[
        #     [
        #         "https://0nism.oss-cn-beijing.aliyuncs.com/home/image-20231130114733931.png",
        #     ],
        # ],
        cache_examples=False,
        title='SAM-TA: Tracking Anything with SAM')

    sam_demo.launch(server_name="0.0.0.0",
                    server_port=12701,
                    share=True)


if __name__ == "__main__":
    main()
