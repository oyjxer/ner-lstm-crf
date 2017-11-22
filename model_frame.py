import tensorflow as tf
import logging


class ner(object):
    def __init__(self, vocb_size, embed_dim, num_units, keep_prob, num_tags, batch_size, max_seq_len, learning_rate,
                 num_epochs, train_set, dev_set, test_set, save_path, min_loss, output_path, id_to_char_dict, id_to_tag_dict):
        self.vocb_size = vocb_size
        self.embed_dim = embed_dim
        self.num_units = num_units
        self.keep_prob = keep_prob
        self.num_tags = num_tags
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.save_path = save_path
        self.min_loss = min_loss
        self.output_path = output_path
        self.id_to_char_dict = id_to_char_dict
        self.id_to_tag_dict = id_to_tag_dict

    def get_batch(self, iter, flag):
        start = iter * self.batch_size
        end = (iter + 1) * self.batch_size
        if flag is "train":
            end = end if end < len(self.train_set[0]) else len(self.train_set[0])
            batch_ids = self.train_set[0][start: end]
            batch_sequence_length = self.train_set[1][start: end]
            batch_tag_indices = self.train_set[2][start: end]
        elif flag is "dev":
            end = end if end < len(self.dev_set[0]) else len(self.train_set[0])
            batch_ids = self.dev_set[0][start: end]
            batch_sequence_length = self.dev_set[1][start: end]
            batch_tag_indices = self.dev_set[2][start: end]
        else:
            end = end if end < len(self.test_set[0]) else len(self.test_set[0])
            batch_ids = self.test_set[0][start: end]
            batch_sequence_length = self.test_set[1][start: end]
            batch_tag_indices = self.test_set[2][start: end]
        return batch_ids, batch_sequence_length, batch_tag_indices

    def build_model(self):
        self.ids = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len])
        self.sequence_length = tf.placeholder(tf.int32, [self.batch_size])
        self.tag_indices = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len])
        params = tf.get_variable("embedding", [self.vocb_size, self.vocb_size])
        embedding = tf.nn.dropout(tf.nn.embedding_lookup(params, self.ids), self.keep_prob)
        cell_fw = tf.contrib.rnn.LSTMCell(self.num_units)
        cell_bw = tf.contrib.rnn.LSTMCell(self.num_units)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding, self.sequence_length,
                                                                    dtype=tf.float32)
        output = tf.reshape(tf.nn.dropout(tf.concat([output_fw, output_bw], 2), self.keep_prob),
                            [-1, self.num_units * 2])
        weight = tf.get_variable("weight", [self.num_units * 2, self.num_tags], tf.float32)
        bias = tf.get_variable("bias", [self.num_tags], tf.float32)
        self.logits = tf.reshape(tf.nn.xw_plus_b(output, weight, bias),
                                 [self.batch_size, self.max_seq_len, self.num_tags])

    def compute_loss(self):
        log_likelihood, self.trainsition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.tag_indices,
                                                                                    self.sequence_length)
        self.loss = tf.reduce_mean(-log_likelihood)

    def train_model(self):
        self.build_model()
        self.compute_loss()
        initializer = tf.global_variables_initializer()
        trainer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(initializer)
            for i in range(self.num_epochs):
                num_batchs = int(len(self.train_set[0]) / self.batch_size)
                for j in range(num_batchs):
                    batch_ids, batch_sequence_length, batch_tag_indices = self.get_batch(j, "train")
                    _, loss = sess.run([trainer, self.loss],
                                       feed_dict={self.ids: batch_ids, self.sequence_length: batch_sequence_length,
                                                  self.tag_indices: batch_tag_indices})
                    logging.info("epoch: %d, iter: %d, loss: %.2f" % (i + 1, j + 1, loss))
                num_batchs = int(len(self.dev_set[0]) / self.batch_size)
                total_loss = 0
                for j in range(num_batchs):
                    batch_ids, batch_sequence_length, batch_tag_indices = self.get_batch(j, "dev")
                    _, loss = sess.run([trainer, self.loss],
                                       feed_dict={self.ids: batch_ids, self.sequence_length: batch_sequence_length,
                                                  self.tag_indices: batch_tag_indices})
                    total_loss += loss
                mean_loss = total_loss / num_batchs
                if mean_loss < self.min_loss:
                    logging.info("mean loss less min loss in dev set, early stop")
                    saver.save(sess, self.save_path)
                    return
            saver.save(sess, self.save_path)

    def test_model(self):
        self.build_model()
        self.compute_loss()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            output_file = tf.gfile.Open(self.output_path, 'wb')
            saver.restore(sess, self.save_path)
            num_batchs = int(len(self.test_set[0]) / self.batch_size)
            for i in range(num_batchs):
                batch_ids, batch_sequence_length, batch_tag_indices = self.get_batch(i, "test")
                logits, trainsition_params = sess.run([self.logits, self.trainsition_params],
                                                      feed_dict={self.ids: batch_ids,
                                                                 self.sequence_length: batch_sequence_length})
                for j in range(self.batch_size):
                    viterbi, _ = tf.contrib.crf.viterbi_decode(logits[j], trainsition_params)
                    for k in range(batch_sequence_length[j]):
                        output_file.write(
                            self.id_to_char_dict[batch_ids[j][k]] + b"\t" + self.id_to_tag_dict[batch_tag_indices[j][k]] + b"\t" +
                            self.id_to_tag_dict[viterbi[k]] + b"\n")
                    output_file.write(b"\n")
            output_file.close()
