import keras
import tensorflow as tf
import vis ## keras-vis
import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers, activations, optimizers, losses, metrics, regularizers

print("keras      {}".format(keras.__version__))
print("tensorflow {}".format(tf.__version__))

from keras.applications.vgg16 import VGG16, preprocess_input
path = '/home/nader/workspace/dal/2019-11-03-21-30-32agent_dq105.h5'
model = keras.models.load_model(path)
# model = VGG16(weights='imagenet')
# model.compile(optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95), loss=losses.mse, metrics=[metrics.mse])
model.summary()

image = open('/home/nader/workspace/dal/res5/2019-34-07-14-34-15_0_682_12_0localview0', 'r')
image = image.readline()
image = eval(image)
image = image['view']
image = np.array(image)
image = np.rot90(image)
image = np.rot90(image)
for x in range(51):
    for y in range(51):
        if 0.29 < image[x][y] < 0.31:
            image[x][y] = 0.0
image = image.reshape((51,51,1))
img               = image
y_pred            = model.predict(img[np.newaxis,...])
class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]

image = image.reshape((51,51))

from vis.utils import utils
layer_idx = utils.find_layer_idx(model, 'dense_4')

# Swap softmax with linear
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)
# help(utils.apply_modifications)

#
from vis.visualization import visualize_saliency
class_idx = class_idxs_sorted[0]
grad_top1 = visualize_saliency(model,
                               layer_idx,
                               filter_indices = class_idx,
                               seed_input     = img[np.newaxis,...])


def show_view(ax,pic):
    s = pic
    gx = []
    gy = []
    ox = []
    oy = []
    tx = []
    ty = []
    sx = []
    sy = []
    for x in range(51):
        for y in range(51):
            if 0.499 < s[x][y] < 0.51:
                gx.append(x)
                gy.append(y)
            elif 0.29 < s[x][y] < 0.31:
                ox.append(x)
                oy.append(y)
            if s[x][y] > 0.9:
                tx.append(x)
                ty.append(y)

    ax.plot(gx, gy, 'o', color='saddlebrown')
    ax.plot(ox, oy, 'o', color='orange')
    ax.plot(tx, ty, 'go')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])

def plot_map(grads):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    show_view(axes[0], image)
    # show_view(axes[1], image)
    # axes[0].imshow(img)
    # axes[1].imshow(img)
    i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
    fig.colorbar(i)

    plt.show()
grad_top1 = np.rot90(grad_top1)
plot_map(grad_top1)