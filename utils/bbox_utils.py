import torch

def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]

    if is_predictions:
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)

        box_predictions[..., 0:2] = torch.sigmoid(
            box_predictions[..., 0:2]
        )

        box_predictions[..., 2:] = torch.exp(
            box_predictions[..., 2:]
        ) * anchors

        scores = torch.sigmoid(predictions[..., 0:1])

        best_class = torch.argmax(
            predictions[..., 5:], dim=-1
        ).unsqueeze(-1)

    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(s)
        .repeat(batch_size, num_anchors, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)

    y = 1 / s * (
        box_predictions[..., 1:2]
        + cell_indices.permute(0, 1, 3, 2, 4)
    )

    width_height = 1 / s * box_predictions[..., 2:4]

    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height),
        dim=-1
    )

    converted_bboxes = converted_bboxes.reshape(
        batch_size,
        num_anchors * s * s,
        6
    )

    return converted_bboxes.tolist()