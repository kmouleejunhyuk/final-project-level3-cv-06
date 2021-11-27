from matplotlib import pyplot as plt
import numpy as np

def draw_batch_images(images, labels, preds, category_names):
    mean, std = 0.5, 0.2
    num_examples = len(images)

    fig, ax = plt.subplots(nrows=num_examples, ncols=1, figsize=(12, 4*num_examples), constrained_layout=True)
    # fig.tight_layout()
    for row_num in range(num_examples):
        # Original Image
        image =  (images[row_num]*std*255) + mean*255

        label = np.where(labels[row_num]==1)[0]
        label = [ category_names[cat_id] for cat_id in label]

        pred = np.where(preds[row_num]==1)[0]
        pred = [ category_names[cat_id] for cat_id in pred]

        ax[row_num].imshow(image.permute(1,2,0))
        ax[row_num].set_title(f"gt : {label},\n pred : {pred}")
    return fig