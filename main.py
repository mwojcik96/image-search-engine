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
    final_rank = sorted(ranking.items(), key=lambda kv: kv[0])
    _, max_val = max(final_rank, key=lambda kv: kv[1])
    final_rank = [(item[0], (max_val - item[1])/max_val) for item in final_rank]
    return final_rank


def parse_query(query):
    query = query.split()
    if len(query) == 3:
        return "max", [query[1]], "max", [query[2]]
    else:
        img_num = int(query[2])
        img_start = 3
        img_stop = img_num + 3
        feat_num = img_stop + 1
        feat_start = feat_num + 1
        return query[1], query[img_start:img_stop], query[img_stop], query[feat_start:]


def calculate_score_for_query(images, names, inv_devs, image_op, image_list, feature_op, feature_list):
    whole_score = None
    for current_image in image_list:

        feature_score = None
        for current_feature in feature_list:
            current_score = simple_process(current_image, current_feature, images, names, inv_devs)
            # print(current_image, current_feature, current_score)

            if feature_score is None:
                feature_score = current_score.copy()
            elif feature_op == 'max':
                feature_score = [(f1[0], max(f1[1], f2[1])) for f1, f2 in zip(feature_score, current_score)]
            elif feature_op == 'min':
                feature_score = [(f1[0], min(f1[1], f2[1])) for f1, f2 in zip(feature_score, current_score)]
            elif feature_op == 'ave':
                feature_score = [(f1[0], f1[1]+f2[1]) for f1, f2 in zip(feature_score, current_score)]

        if feature_op == 'ave':
            feature_score = [(f[0], f[1]/len(feat_list)) for f in feature_score]
        # print(current_image, feat_op, feature_score)

        if whole_score is None:
            whole_score = feature_score.copy()
        elif image_op == 'max':
            whole_score = [(f1[0], max(f1[1], f2[1])) for f1, f2 in zip(whole_score, feature_score)]
        elif image_op == 'min':
            whole_score = [(f1[0], min(f1[1], f2[1])) for f1, f2 in zip(whole_score, feature_score)]
        elif image_op == 'ave':
            whole_score = [(f1[0], f1[1]+f2[1]) for f1, f2 in zip(whole_score, feature_score)]

    if image_op == 'ave':
        whole_score = [(f[0], f[1]/len(img_list)) for f in whole_score]
    # print(img_list, img_op, whole_score)
    final_score = sorted(whole_score, key=lambda kv: kv[1], reverse=True)
    return final_score


def print_scores(scores_list, num):
    for item in scores_list[:num]:
        print(item[0], item[1])


if __name__ == "__main__":
    images, names, inv_deviations = prepare_dataset()
    print("Enter 'exit' to end")
    query = ''
    while 1:
        query = input("Enter your query: ")
        if query == 'exit':
            break
        img_op, img_list, feat_op, feat_list = parse_query(query)
        score = calculate_score_for_query(images, names, inv_deviations, img_op, img_list, feat_op, feat_list)
        print_scores(score, 12)
