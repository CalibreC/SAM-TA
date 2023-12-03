from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SAM_predictor:
    def __init__(self, cfg: dict):
        self.model_type = cfg["model_type"]
        self.checkpoint_path = cfg["checkpoint_path"]
        self.device = cfg["device"]

        sam_model_reg = sam_model_registry[self.model_type]
        self.sam = sam_model_reg(checkpoint=self.checkpoint_path).to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def generator_masks(self, image):
        sam_result = self.mask_generator.generate(image)

        return sam_result
