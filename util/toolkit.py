import operator


def get_unique_values(array):
    print(array.size)
    counter = 0
    counter2 = 0
    dict = {}
    for sentence in array:
        counter = counter + 1
        counter2 = counter2 + sentence.size
        for word in sentence:
            if dict.get(word, "") == "":
                dict[word] = 1
            else:
                dict[word] = dict[word] + 1

    # print(dict)
    return dict


def dict_get_max(dict):
    return max(dict.items(), key=operator.itemgetter(1))[0]


def unnest_twofold_array(array):
    unfolded = []
    for elem in array:
        for subelem in elem:
            # print(subelem)
            # print(type(subelem))
            unfolded.append(subelem)

    return unfolded
