from matplotlib import pyplot as plt
import numpy as np

def draw_batch_images(images, labels, preds, category_names):
    mean, std = 0.5, 0.2
    num_examples = len(images)
    n_cols = 4
    fig, axes = plt.subplots(
        nrows= int(np.ceil(num_examples / n_cols)), 
        ncols=n_cols, 
        figsize=(30, int(4*num_examples / n_cols)), 
        constrained_layout=True
    )

    # fig.tight_layout()
    for row_num, ax in zip(range(num_examples), axes.ravel()):
        # Original Image
        image =  (images[row_num]*std*255) + mean*255

        label = np.where(labels[row_num]==1)[0]
        label = [ category_names[cat_id] for cat_id in label]

        pred = np.where(preds[row_num]==1)[0]
        pred = [ category_names[cat_id] for cat_id in pred]

        ax.imshow(image.permute(1,2,0))
        ax.set_title(f"gt : {label},\n pred : {pred}")
    return fig