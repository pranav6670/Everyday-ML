import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm # sklearn = scikit-learn
from sklearn.datasets import make_moons
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

X, Y = make_moons(100, noise=0.1, random_state=2)  # semi-random data

fig, ax = plt.subplots(1, figsize=(5, 5), facecolor=(1, 1, 1))
fig.subplots_adjust(left=0, right=1, bottom=0)
xx, yy = np.meshgrid(np.linspace(-2, 3, 500), np.linspace(-1, 2, 500))


def make_frame(t):
    ax.clear()
    ax.axis('off')
    ax.set_title("SVM classification", fontsize=16)

    classifier = svm.SVC(gamma=2, C=1)
    # the varying weights make the points appear one after the other
    weights = np.minimum(1, np.maximum(0, t**2+10-np.arange(100)))
    classifier.fit(X, Y, sample_weight=weights)
    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.8,
                vmin=-2.5, vmax=2.5, levels=np.linspace(-2, 2, 20))
    ax.scatter(X[:, 0], X[:, 1], c=Y, s=100*weights, cmap=plt.cm.bone)

    return mplfig_to_npimage(fig)


animation = VideoClip(make_frame, duration=20)
animation.write_videofile("svm.mp4", fps=40)
