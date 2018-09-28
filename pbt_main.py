import argparse
import numpy as np
import sys
import time
import tensorflow as tf

from gan_class import *

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    # we need to provide all ps and worker info to each server so they are aware of each other
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    
    # create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    # create and start a server for the local task.
    server = tf.train.Server(cluster,
                            job_name=FLAGS.job_name,
                            task_index=FLAGS.task_index)
                            
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # explictely place weights and hyperparameters on the worker servers to prevent sharing
        # otherwise replica_device_setter will put them on the ps
        with tf.device("/job:worker/task:{}".format(FLAGS.task_index)):

            # [step 1] initialize all worker variables
            # pbt-hyperparams and weights MUST go here
            gan = GAN(worker_idx=FLAGS.task_index, epochs=50)

            # [step 2] define graph here
            # we don't define on the ps because we don't share weights
            gan.build_model()

            gan.saver = tf.train.Saver(max_to_keep=1)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                            is_chief=True) as gan.mon_sess:

            ready_freq = 100

            for epoch in range(gan.epochs):
                for idx in range(gan.num_batches):

                    gan.step(idx, epoch)

                    # inception score takes ~5s, so there is a tradeoff
                    if idx % ready_freq == 0:
                        inception_score, _ = gan.eval()
                        print("Worker {} with Inception Score {}".format(FLAGS.task_index, inception_score))

                        do_explore = gan.exploit(worker_idx=FLAGS.task_index, score=inception_score)

                        if do_explore:
                            gan.explore()
                            inception_score, _ = gan.eval()
                            print("Worker {} with Inception Score {}".format(FLAGS.task_index, inception_score))

                    if idx % 5 == 0:
                        # update checkpoint (ideally checkpoint every idx)
                        gan.save(worker_idx=FLAGS.task_index, score=inception_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

