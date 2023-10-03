from create_features import mark_offensive_words

features_file = 'train.tsv'
filename1 = 'predicted_labels/bert-keep-1_test.txt'
filename2 = 'predicted_labels/bert-replace-1_test.txt'
filename3 = 'predicted_labels/bert-remove-1_test.txt'


def read_file(filename):
    with open(filename, encoding='utf-8') as file:
        rows = []
        for row in file:
            rows.append(row)
    return rows


features = read_file(features_file)
labels1 = read_file(filename1)
labels2 = read_file(filename2)
labels3 = read_file(filename3)

for idx in range(len(labels1)):
    if labels1[idx] != labels2[idx] or labels1[idx] != labels3[idx]:
        print(features[idx])
        print(mark_offensive_words(features[idx]), 'model 1 predicted: ' + labels1[idx],
              'model 2 predicted:' + labels2[idx], 'model 3 predicted:' + labels3[idx])
