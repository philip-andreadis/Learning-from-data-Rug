from statistics import *


def get_results(results):
    f1 = []
    precision = []
    recall = []
    accuracy = []
    for result in results:
        items = result.split()
        f1.append(float(items[0]))
        precision.append(float(items[2]))
        recall.append(float(items[4]))
        accuracy.append(float(items[6]))

    result_str = '${:.3f} \\pm {:.3f}$'.format(mean(f1), stdev(f1))
    result_str += ' & ${:.3f} \\pm {:.3f}$'.format(mean(precision), stdev(precision))
    result_str += ' & ${:.3f} \\pm {:.3f}$'.format(mean(recall), stdev(recall))
    result_str += ' & ${:.3f} \\pm {:.3f}$ \\\\'.format(mean(accuracy), stdev(accuracy))
    print(result_str)


with open('results.txt') as file:
    results = []
    for row in file:
        if 'f1-score' not in row:
            results.append(row)

dev_results = [results[0], results[2], results[4]]
test_results = [results[1], results[3], results[5]]

get_results(dev_results)
get_results(test_results)
