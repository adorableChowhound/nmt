from nmt.nmt import *

FLAGS = None
data_dir = 'data/jp_zh_data'
model_dir = 'model/jp2zh_model'


def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    train_fn = train.train
    inference_fn = inference.inference
    run_main(FLAGS, default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
    sys.argv = ['nmt_main.py', '--src=jp', '--tgt=zh', '--vocab_prefix=' + data_dir + '/vocab',
                '--train_prefix=' + data_dir + '/train',
                '--dev_prefix=' + data_dir + '/dev', '--test_prefix=' + data_dir + '/test',
                '--out_dir=' + model_dir,
                '--num_train_steps=17000', '--steps_per_stats=100', '--num_layers=2', '--num_units=128',
                '--dropout=0.2', '--metrics=bleu']
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    # print(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    # print(unparsed)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)