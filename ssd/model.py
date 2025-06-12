from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights, _utils
from torchvision.models.detection.ssd import SSDClassificationHead

def construct_ssd_model(score_thresh=0.01, nms_thresh=0.45):
    """
    Constructs an SSD model with a custom classification head for the specific task (parking space detection)
    """

    # Load the pre-trained SSD300 model with VGG16 weights
    model = ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1,
        progress=True,
    )

    # get the model classification head parameters
    in_channels = _utils.retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    num_classes = 3 # 2 classes + background

    # Create a new classification head with the correct parameters
    new_classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    # set the new classification head to the model
    model.head.classification_head = new_classification_head

    return model