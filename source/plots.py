import matplotlib.pyplot as plt
import os


def accuracy_loss_plots(history, model_name: str, save_dir: str) -> None:
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()

    plt.savefig(os.path.join(save_dir, model_name + "_accuracy_loss.png"))
    plt.close()


def save_accuracy_loss(history, model_name: str, save_dir: str):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = len(acc)

    with open(os.path.join(save_dir, model_name + "_accuracy_loss.csv"), "w") as f:
        f.write("epoch,acc,val_acc,loss,val_loss\n")
        for i in range(epochs):
            f.write(f"{i},{acc[i]},{val_acc[i]},{loss[i]},{val_loss[i]}\n")
