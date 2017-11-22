from model_frame import ner
from load_data import get_dict, get_train_dev_data
import argparse, os


def train_model(data_dir, model_dir):
    char_to_id_dict, tag_to_id_dict, id_to_char_dict, id_to_tag_dict = get_dict(os.path.join(data_dir, "chars.txt"),
                                                                                os.path.join(data_dir, "tags.txt"))
    train_set, dev_set = get_train_dev_data(char_to_id_dict, tag_to_id_dict,
                                            os.path.join(data_dir, "train_dev_data.txt"), 50)
    # vocb_size, embed_dim, num_units, keep_prob, num_tags, batch_size, max_seq_len, learning_rate, num_epochs, train_set, dev_set, test_set, save_path, min_loss, output_path, char_dict, tag_dict
    model = ner(len(id_to_char_dict), 64, 64, 0.01, len(id_to_tag_dict), 20, 50, 0.01, 10000, train_set, dev_set, None,
                os.path.join(model_dir, "model.ckpt"), 0.001, None, None, None)
    model.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_dir", type=str)
    flags, _ = parser.parse_known_args()
    train_model(flags.data_dir, flags.model_dir)
