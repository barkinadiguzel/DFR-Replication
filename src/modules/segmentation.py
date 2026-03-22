import torch.nn.functional as F


def segment_anomaly(anomaly_map, output_size, threshold):
    anomaly_map = F.interpolate(anomaly_map, size=output_size, mode="bilinear", align_corners=False)

    mask = (anomaly_map > threshold).float()
    return anomaly_map, mask
