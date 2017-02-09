from datetime import datetime
import os
import shutil
import sys

# def plot_generated_batch(X_real, generator_model, batch_size, noise_dim, image_dim_ordering, noise_scale=0.5):
#
#     # Generate images
#     X_gen = sample_noise(noise_scale, batch_size, noise_dim)
#     X_gen = generator_model.predict(X_gen)
#
#     X_real = inverse_normalization(X_real)
#     X_gen = inverse_normalization(X_gen)
#
#     Xg = X_gen[:8]
#     Xr = X_real[:8]
#
#     if image_dim_ordering == "tf":
#         X = np.concatenate((Xg, Xr), axis=0)
#         list_rows = []
#         for i in range(int(X.shape[0] / 4)):
#             Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
#             list_rows.append(Xr)
#
#         Xr = np.concatenate(list_rows, axis=0)
#
#     if image_dim_ordering == "th":
#         X = np.concatenate((Xg, Xr), axis=0)
#         list_rows = []
#         for i in range(int(X.shape[0] / 4)):
#             Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
#             list_rows.append(Xr)
#
#         Xr = np.concatenate(list_rows, axis=1)
#         Xr = Xr.transpose(1,2,0)
#
#     if Xr.shape[-1] == 1:
#         plt.imshow(Xr[:, :, 0], cmap="gray")
#     else:
#         plt.imshow(Xr)
#     plt.savefig("../../figures/current_batch.png")
#     plt.clf()
#     plt.close()

def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.','-')[:-7]


def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def clear_dir(directory):
    """
    Removes all files in the given directory.

    @param directory: The path to the directory.
    """
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)


# add path to search paths
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
