import argparse
from glob import glob
from time import sleep

import imagehash
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.externals import joblib
from tqdm import tqdm


def hamming(str1, str2):
    return sum(map(str.__ne__, str1, str2))


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser(description='Returns an image dataset with near-duplicate images removed.')
    parser.add_argument('load_path', type=str, help='File search pattern to find images (glob style).')
    parser.add_argument('save_path', type=str, help='Location to save the pickled list of unique files.')
    parser.add_argument('-hd', '--min_hamming_distance', type=int, default=10,
                        help='Minimum hamming distance needed to be considered unique.')
    parser.add_argument('-lm', '--limit', type=int, default=None, help='Only run on first n files.')
    args = parser.parse_args()

    # Get files
    files = sorted(glob(args.load_path))
    if args.limit is not None:
        files = files[:args.limit]
    n_files = len(files)
    print('Analyzing {} images...'.format(n_files))

    hashes_seen = set()
    unique_files = []

    sleep(.01)
    for image_path in tqdm(files):
        # Get image hash
        image = Image.open(image_path)
        h = str(imagehash.phash(image, hash_size=8))

        # See if it is unique
        unique = True
        for seen_hash in hashes_seen:
            if hamming(seen_hash, h) <= args.min_hamming_distance:
                unique = False
                break
        if unique:
            hashes_seen.add(h)
            unique_files.append(image_path)
    sleep(.01)

    # Print diagnostics and save
    n_unique_files = len(unique_files)
    print('\n{} unique images found ({:.1f}% of original set).'.format(n_unique_files, n_unique_files / n_files * 100))
    print('Saving list of file names to {}).'.format(args.save_path))
    joblib.dump(unique_files, args.save_path)

    # Plot origonal set
    nrows, ncols = 6, 10
    fig, axes = plt.subplots(nrows, ncols)
    plt.suptitle('Original Set (first {})'.format(len(axes.flatten())))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(plt.imread(files[i]))
        ax.set_title(files[i].split('.')[-2][-4:])
        ax.axis('off')

    # Plot filtered set
    fig, axes = plt.subplots(nrows, ncols)
    plt.suptitle('Filtered Set (first {})'.format(len(axes.flatten())))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(plt.imread(unique_files[i]))
        ax.set_title(unique_files[i].split('.')[-2][-4:])
        ax.axis('off')

    plt.show()
