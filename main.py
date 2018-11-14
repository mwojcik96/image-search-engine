import os
import numpy as np
from image import Image


def prepare_dataset():
    images_dir = './images'
    features_dir = './features'
    image_list = list()
    name_list = list()
    inv_std_devs = dict()
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            name_list.append(filename)
            image_list.append(Image(filename))
    for featurename in os.listdir(features_dir):
        if featurename.endswith(".dat"):
            with open(os.path.join(features_dir, featurename)) as file:
                deviations = None
                for line in file:
                    splited_line = line.split()
                    if deviations is None:
                        deviations = np.empty((0, len(splited_line)-1))
                    image_list[name_list.index(splited_line[0])].fill_measure(
                        os.path.splitext(featurename)[0], [float(it) for it in splited_line[1:]])
                    deviations = np.append(deviations, [splited_line[1:]], axis=0)
                deviations = np.std(deviations.astype(np.float), axis=0)
                inv_std_devs[os.path.splitext(featurename)[0]] = 1/deviations
    return image_list, name_list, inv_std_devs


def simple_process(filename, featurename, imglist, namelist, invstddevs):
    index = namelist.index(filename)
    query_image = imglist[index]
    ranking = dict()
    for img in imglist:
        ranking[img.filename] = np.sum(np.multiply(invstddevs[featurename], np.absolute(
                np.array(img.features[featurename]) - np.array(query_image.features[featurename]))))
    final_rank = sorted(ranking.items(), key=lambda kv: kv[1])
    max_val = final_rank[-1][1]
    final_rank = [(item[0], (max_val - item[1])/max_val) for item in final_rank]
    return final_rank


if __name__ == "__main__":
    images, names, inv_deviations = prepare_dataset()
    query = ['']
    while query[0] != 'findimage':
        query = input("Enter your query:").split()
    f_name = query[1]
    feature = query[2]
    rank = simple_process(f_name, feature, images, names, inv_deviations)
    print("RANKING:")
    for item in rank:
        print(item[0], item[1])