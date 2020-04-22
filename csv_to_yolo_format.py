import csv
import os
import sys

def fit_scale(xmin, ymin, xmax, ymax, ORIG=(720, 1280), TRAIN=(320, 640)):
    h_orig, w_orig = ORIG
    h_train, w_train = TRAIN
    h_scale = h_train/h_orig
    w_scale = w_train/w_orig    

    xmin = int(xmin*w_scale)
    xmax = int(xmax*w_scale)
    ymin = int(ymin*h_scale)
    ymax = int(ymax*h_scale)

    return xmin, ymin, xmax, ymax


def main(FILEPATH):
    print(os.path.abspath(FILEPATH))

    DIR_NAME = os.path.dirname(
        os.path.abspath(FILEPATH)
        )

    with open(FILEPATH, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    image_dict = dict()

    for row in data[1:]:
        row.append(0) # class label
        tmp = image_dict.get(row[0], [])
        tmp.append(row[4:])
        image_dict[row[0]] = tmp

    row_list = []

    for key in image_dict.keys():
        bbs = image_dict[key]
        row_text = os.path.join(DIR_NAME, key)
        for bb in bbs:
            xmin, ymin, xmax, ymax, c = bb
            xmin, ymin, xmax, ymax = fit_scale(int(xmin),int(ymin),int(xmax),int(ymax))
            row_text = row_text + " {},{},{},{},{}".format(xmin, ymin, xmax, ymax, c)
        # print(row_text)
        row_list.append(row_text)

    train_txt = '\n'.join(row_list)

    print(train_txt)

    with open(FILEPATH, newline='') as f:
        f = open("output.txt", 'w')
        f.write(train_txt)
        f.close()

if __name__ == "__main__":
    args = sys.argv
    main(args[1])