import tensorflow as tf


def get_dict(chars_path, tags_path):
    char_to_id_dict = {}
    tag_to_id_dict = {}
    i = 2
    with tf.gfile.Open(chars_path, "rb") as chars_file:
        for line in chars_file.readlines():
            if not line.strip() in char_to_id_dict:
                char_to_id_dict[line.strip()] = i
                i = i + 1
    j = 0
    with tf.gfile.Open(tags_path, "rb") as tags_file:
        for line in tags_file.readlines():
            if not line.strip() in tag_to_id_dict:
                tag_to_id_dict[line.strip()] = j
                j = j + 1
    id_to_char_dict = {0: b"unk", 1: b"pad"}
    id_to_tag_dict = {}
    for char in char_to_id_dict.keys():
        id_to_char_dict[char_to_id_dict[char]] = char
    for tag in tag_to_id_dict.keys():
        id_to_tag_dict[tag_to_id_dict[tag]] = tag
    return char_to_id_dict, tag_to_id_dict, id_to_char_dict, id_to_tag_dict


def get_test_data(char_to_id_dict, tag_to_id_dict, data_path, max_time):
    test_chars = []
    test_seq_len = []
    test_tags = []
    temp_chars = []
    temp_tags = []
    with tf.gfile.Open(data_path, "rb") as data_file:
        for line in data_file.readlines():
            if line.strip() is b"":
                temp_seq_len = len(temp_chars)
                if temp_seq_len < max_time:
                    temp_chars += [1] * (max_time - len(temp_chars))
                    temp_tags += [0] * (max_time - len(temp_tags))
                elif temp_seq_len > max_time:
                    temp_chars = temp_chars[0: max_time]
                    temp_tags = temp_tags[0: max_time]
                    temp_seq_len = max_time
                test_chars.append(temp_chars)
                test_seq_len.append(temp_seq_len)
                test_tags.append(temp_tags)
                temp_chars = []
                temp_tags = []
                continue
            char, tag = line.strip().split(b"\t")
            temp_chars.append(char_to_id_dict[char] if char in char_to_id_dict else 0)
            temp_tags.append(tag_to_id_dict[tag])
    return (test_chars, test_seq_len, test_tags)


def get_train_dev_data(char_to_id_dict, tag_to_id_dict, data_path, max_time):
    train_dev_chars = []
    train_dev_seq_len = []
    train_dev_tags = []
    temp_chars = []
    temp_tags = []
    with tf.gfile.Open(data_path, "rb") as data_file:
        for line in data_file.readlines():
            if line.strip() is b"":
                temp_seq_len = len(temp_chars)
                if temp_seq_len < max_time:
                    temp_chars += [1] * (max_time - len(temp_chars))
                    temp_tags += [0] * (max_time - len(temp_tags))
                elif temp_seq_len > max_time:
                    temp_chars = temp_chars[0: max_time]
                    temp_tags = temp_tags[0: max_time]
                    temp_seq_len = max_time
                train_dev_chars.append(temp_chars)
                train_dev_seq_len.append(temp_seq_len)
                train_dev_tags.append(temp_tags)
                temp_chars = []
                temp_tags = []
                continue
            char, tag = line.strip().split(b"\t")
            temp_chars.append(char_to_id_dict[char] if char in char_to_id_dict else 0)
            temp_tags.append(tag_to_id_dict[tag])
    length = len(train_dev_chars)
    threshold = int(length * 0.6)
    return (train_dev_chars[0: threshold], train_dev_seq_len[0: threshold], train_dev_tags[0: threshold]), (
        train_dev_chars[threshold: length], train_dev_seq_len[threshold: length], train_dev_tags[threshold: length])
