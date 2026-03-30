import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_image(image, boxes, class_labels):
    colour_map = plt.get_cmap("tab20b")
    colors = [
        colour_map(i)
        for i in np.linspace(0, 1, len(class_labels))
    ]

    img = np.array(image)
    h, w, _ = img.shape

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)

    for box in boxes:
        class_pred = int(box[0])
        box = box[2:]

        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * w, upper_left_y * h),
            box[2] * w,
            box[3] * h,
            linewidth=2,
            edgecolor=colors[class_pred],
            facecolor="none"
        )
        ax.add_patch(rect)
        plt.text(
            upper_left_x * w,
            upper_left_y * h,
            s=class_labels[class_pred],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[class_pred], "pad": 0}
        )
    plt.show()
