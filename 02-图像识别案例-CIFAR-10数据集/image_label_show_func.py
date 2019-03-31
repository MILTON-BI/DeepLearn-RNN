"""显示图像和对应标签的函数"""
import matplotlib.pyplot as plt
# 定义标签字典，每个数字代表相应标签的名称
label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3:'cat', 4: 'deer', 5: 'dog', 6:'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    if num > 10:
        num = 10
    for i in range(0, num):
        ax = plt.subplot(2, 5, i+1)
        ax.imshow(images[idx], cmap='binary')

        title = str(i) + ',' + label_dict[labels[idx]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[idx]]

        ax.set_title(title, fontsize=10)

        idx += 1
    plt.show()