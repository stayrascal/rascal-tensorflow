
1. tf.merge_all_summaries()        => tf.summary.merge_all()
2. tf.scalar_summary('loss', loss) => tf.summary.scalar('loss', loss)
3. tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels,name='xentropy')
    => tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
4. tf.initialize_all_variables() => tf.global_variables_initializer()
5. tf.train.SummaryWriter(FLAGS.train_dir,graph_def=sess.graph_def)
    => tf.summary.FileWriter(FLAGS.train_dir, graph_def=sess.graph_def)