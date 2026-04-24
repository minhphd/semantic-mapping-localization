import torch
from ultralytics import YOLO


class YOLODetector():
    """
    YOLOv11/YOLOv8 for detection only.
    Returns:
        bboxes   (N, 4)
        class_id (N,)
        conf     (N,)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.segmentation["detector"] == "yolov11":
            ckpt = cfg.paths["yolo11_ckpt"]
        elif cfg.segmentation["detector"] == "yolov8":
            ckpt = cfg.paths["yolo8_ckpt"]
        else:
            raise ValueError(f"Invalid detector: {cfg.segmentation['detector']}")
        print(f"[YOLO Detector] Loading {ckpt}")

        self.model = YOLO(ckpt)
        self.model.to(cfg.device)

        # Set open-world class names
        self.class_names = cfg.landmarks["classes"]
        self.model.set_classes(self.class_names)

    def __call__(self, rgb):
        """rgb: np.ndarray (H,W,3)"""
        results = self.model(rgb, device=self.cfg.device, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return [], [], []

        b = results.boxes
        bboxes = b.xyxy.cpu().numpy()
        class_ids = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()

        return bboxes, class_ids, conf
