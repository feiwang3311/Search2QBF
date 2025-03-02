from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import random
import os
import time
from problems_loader import init_problems_loader
from mlp import MLP
from util import repeat_end, decode_final_reducer, decode_transfer_fn
from tensorflow.contrib.rnn import LSTMStateTuple
#from sklearn.cluster import KMeans
from tensorflow_ranking import losses

class NeuroSAT(object):
    def __init__(self, opts):
        self.opts = opts
        self.final_reducer = decode_final_reducer(opts.final_reducer)
        self.build_network()
        self.train_problems_loader = None
        self.train_counter = 0
        self.epoch_size = opts.epoch_size
        self.epoch = 0

    def init_random_seeds(self):
        tf.set_random_seed(self.opts.tf_seed)
        np.random.seed(self.opts.np_seed)

    def construct_session(self):
        self.sess = tf.Session()

    def declare_parameters(self):
        opts = self.opts
        with tf.variable_scope('params') as scope:
            self.A_init = tf.get_variable(name="A_init", initializer=tf.random_normal([1, self.opts.d]))
            self.L_init = tf.get_variable(name="L_init", initializer=tf.random_normal([1, self.opts.d]))
            self.C_init = tf.get_variable(name="C_init", initializer=tf.random_normal([1, self.opts.d]))

            self.A_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("A_msg"))
            self.L_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("L_msg"))
            self.C_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("C_msg"))

            self.A_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))
            self.L_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))
            self.C_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))
            self.C_update2 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))

            #self.A_vote = MLP(opts, opts.d, repeat_end(opts.d, opts.n_vote_layers, 1), name=("A_vote"))
            #self.L_vote = MLP(opts, opts.d, repeat_end(opts.d, opts.n_vote_layers, 1), name=("L_vote"))
            #self.vote_bias = tf.get_variable(name="vote_bias", shape=[], initializer=tf.zeros_initializer())
            self.A_filter = MLP(opts, opts.d * 2, repeat_end(opts.d, opts.n_filter_layers, opts.n_filter_d), name=("A_filter"))
            self.A_grader = MLP(opts, opts.n_filter_d, [1], name=("A_grader"))

    def declare_placeholders(self):
        self.n_A_vars = tf.placeholder(tf.int32, shape=[], name='n_A_vars')
        self.n_A_lits = tf.placeholder(tf.int32, shape=[], name='n_A_lits')
        self.n_L_vars = tf.placeholder(tf.int32, shape=[], name='n_L_vars')
        self.n_L_lits = tf.placeholder(tf.int32, shape=[], name='n_L_lits')
        self.n_clauses = tf.placeholder(tf.int32, shape=[], name='n_clauses')
        self.n_A_vars_per_problem = tf.placeholder(tf.int32, shape=[], name='n_A_vars_per_problem')
        self.n_L_vars_per_problem = tf.placeholder(tf.int32, shape=[], name='n_L_vars_per_problem')

        self.A_unpack = tf.sparse_placeholder(tf.float32, shape=[None, None], name='A_unpack')
        self.L_unpack = tf.sparse_placeholder(tf.float32, shape=[None, None], name='L_unpack')
        # self.is_sat = tf.placeholder(tf.bool, shape=[None], name='is_sat')

        # useful helpers
        self.n_batches = tf.div(self.n_A_vars, self.n_A_vars_per_problem)
        # self.n_batches = tf.placeholder(tf.int32, shape=[], name='n_batches')

        # candidates to be ranked by the neural network (None represent number of candidates for each problem)
        self.candidates = tf.placeholder(tf.float32, shape=[None, None, None], name='candidates')
        # labels: A `Tensor` of shape [batchSize, candidateSize] representing relevance.
        self.labels = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    def while_cond(self, i, L_state, C_state, A_state):
        return tf.less(i, self.opts.n_rounds)

    def flip(self, lits, size):
        return tf.concat([lits[size : (2 * size), :], lits[0 : size, :]], axis=0)

    def while_body(self, i, L_state, C_state, A_state):
        A_pre_msgs = self.A_msg.forward(A_state.h)
        AC_msgs = tf.sparse_tensor_dense_matmul(self.A_unpack, A_pre_msgs, adjoint_a=True)

        L_pre_msgs = self.L_msg.forward(L_state.h)
        LC_msgs = tf.sparse_tensor_dense_matmul(self.L_unpack, L_pre_msgs, adjoint_a=True)

        # maybe have 2 C_update modules, 1 for LC_msgs and 1 for AC_msgs
        with tf.variable_scope('C_update') as scope:
            _, C_state = self.C_update(inputs= LC_msgs, state=C_state)
        with tf.variable_scope('C_update2') as scope:
            _, C_state = self.C_update2(inputs=AC_msgs, state=C_state)

        C_pre_msgs = self.C_msg.forward(C_state.h)
        CL_msgs = tf.sparse_tensor_dense_matmul(self.L_unpack, C_pre_msgs)
        CA_msgs = tf.sparse_tensor_dense_matmul(self.A_unpack, C_pre_msgs)

        with tf.variable_scope('L_update') as scope:
            _, L_state = self.L_update(inputs=tf.concat([CL_msgs, self.flip(L_state.h, self.n_L_vars)], axis=1), state=L_state)
        with tf.variable_scope('A_update') as scope:
            _, A_state = self.A_update(inputs=tf.concat([CA_msgs, self.flip(A_state.h, self.n_A_vars)], axis=1), state=A_state)

        return i+1, L_state, C_state, A_state

    def pass_messages(self):
        with tf.name_scope('pass_messages') as scope:
            denom = tf.sqrt(tf.cast(self.opts.d, tf.float32))

            A_output = tf.tile(tf.div(self.A_init, denom), [self.n_A_lits, 1])
            L_output = tf.tile(tf.div(self.L_init, denom), [self.n_L_lits, 1])
            C_output = tf.tile(tf.div(self.C_init, denom), [self.n_clauses, 1])

            A_state = LSTMStateTuple(h=A_output, c=tf.zeros([self.n_A_lits, self.opts.d]))
            L_state = LSTMStateTuple(h=L_output, c=tf.zeros([self.n_L_lits, self.opts.d]))
            C_state = LSTMStateTuple(h=C_output, c=tf.zeros([self.n_clauses, self.opts.d]))

            _, L_state, C_state, A_state = tf.while_loop(self.while_cond, self.while_body, [0, L_state, C_state, A_state])

        self.final_A_lits = A_state.h
        # self.final_L_lits = L_state.h
        # self.final_clauses = C_state.h
        
    def compute_grades(self):
        with tf.name_scope('compute_grades') as scope:
            self.final_A_vars = tf.concat([self.final_A_lits[0:self.n_A_vars], self.final_A_lits[self.n_A_vars:self.n_A_lits]], axis=1)
            self.final_A_filter = self.A_filter.forward(self.final_A_vars)
            self.final_A_filter_batched = tf.reshape(self.final_A_filter, [self.n_batches, self.n_A_vars_per_problem, self.opts.n_filter_d])
            self.final_pre_grades = tf.matmul(self.candidates, self.final_A_filter_batched) # n_batch x None x opts.n_filter_d (None is candidate size)
            self.final_pre_grades_reshaped = tf.reshape(self.final_pre_grades, [-1, self.opts.n_filter_d])
            self.final_grades = self.A_grader.forward(self.final_pre_grades_reshaped) # None x 1 (None is candidate size * n_batch)
            self.grades = tf.reshape(self.final_grades, [self.n_batches, -1])

    def compute_ranking_loss(self):
        self.loss_fn = losses.make_loss_fn('pairwise_logistic_loss', lambda_weight=losses.create_ndcg_lambda_weight(), name='ranking_loss_fn')
        self.ranking_loss = self.loss_fn(self.labels, self.grades, None)
        
        with tf.name_scope('l2') as scope:
            l2_cost = tf.zeros([])
            for var in tf.trainable_variables():
                l2_cost += tf.nn.l2_loss(var)

        self.cost = tf.identity(self.ranking_loss + self.opts.l2_weight * l2_cost, name="cost")

    def build_optimizer(self):
        opts = self.opts

        self.global_step = tf.get_variable("global_step", shape=[], initializer=tf.zeros_initializer(), trainable=False)

        if opts.lr_decay_type == "no_decay":
            self.learning_rate = tf.constant(opts.lr_start)
        elif opts.lr_decay_type == "poly":
            self.learning_rate = tf.train.polynomial_decay(opts.lr_start, self.global_step, opts.lr_decay_steps, opts.lr_end, power=opts.lr_power)
        elif opts.lr_decay_type == "exp":
            self.learning_rate = tf.train.exponential_decay(opts.lr_start, self.global_step, opts.lr_decay_steps, opts.lr_decay, staircase=False)
        else:
            raise Exception("lr_decay_type must be 'no_decay', 'poly' or 'exp'")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, self.opts.clip_val)
        self.apply_gradients = optimizer.apply_gradients(zip(gradients, variables), name='apply_gradients', global_step=self.global_step)

    def initialize_vars(self):
        tf.global_variables_initializer().run(session=self.sess)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.opts.n_saves_to_keep)
        if self.opts.run_id is not None:
            self.save_dir = "snapshots/run%d" % self.opts.run_id
            self.save_prefix = os.path.join(self.save_dir, "snap") #"%s/snap" % self.save_dir

    def build_network(self):
        self.init_random_seeds()
        self.construct_session()
        self.declare_parameters()
        self.declare_placeholders()
        self.pass_messages()
        # self.compute_logits()
        self.compute_grades()
        # self.compute_cost()
        self.compute_ranking_loss()
        self.build_optimizer()
        self.initialize_vars()
        self.init_saver()

    def save(self, epoch):
        self.saver.save(self.sess, self.save_prefix, global_step=epoch)

    def restore(self):
        snapshot = "snapshots/run%d/snap-%d" % (self.opts.restore_id, self.opts.restore_epoch)
        self.saver.restore(self.sess, snapshot)

    def build_feed_dict(self, problem, candidates, labels):
        d = {}
        d[self.n_A_vars] = problem.n_vars_AL[0]
        d[self.n_A_lits] = problem.n_lits_AL[0]
        d[self.n_L_vars] = problem.n_vars_AL[1]
        d[self.n_L_lits] = problem.n_lits_AL[1]
        d[self.n_clauses] = problem.n_clauses
        d[self.n_A_vars_per_problem] = problem.sizes[0]
        d[self.n_L_vars_per_problem] = problem.sizes[1]

        d[self.L_unpack] =  tf.SparseTensorValue(indices=problem.L_unpack_indices,
                                                 values=np.ones(problem.L_unpack_indices.shape[0]),
                                                 dense_shape=[problem.n_lits_AL[1], problem.n_clauses])
        d[self.A_unpack] =  tf.SparseTensorValue(indices=problem.A_unpack_indices,
                                                 values=np.ones(problem.A_unpack_indices.shape[0]),
                                                 dense_shape=[problem.n_lits_AL[0], problem.n_clauses])

        #d[self.is_sat] = problem.is_sat
        d[self.candidates] = candidates
        d[self.labels] = labels

        return d

    def train_one(self, problem, candidates, labels):
        d = self.build_feed_dict(problem, candidates, labels)
        _, grades, cost = self.sess.run([self.apply_gradients, self.grades, self.cost], feed_dict=d)
        self.train_counter += 1
        if self.train_counter >= self.epoch_size:
            self.train_counter = 0
            self.epoch += 1
            self.save(epoch)
        return grades, cost

    def inference(self, problem, candidates):
        d = self.build_feed_dict(problem, candidates, np.zeros((1,1),dtype=np.float32))
        grades = self.sess.run(self.grades, feed_dict=d)
        return grades

    def filter(self, problem, candidates, top_k):
        grades = self.inference(problem, candidates)
        index = np.argsort(grades)
        filtered_can = np.array([candidates[i][index[i][::-1][:top_k]] for i in range(index.shape[0])])
        return filtered_can

    def train_epoch(self, epoch):
        if self.train_problems_loader is None:
            self.train_problems_loader = init_problems_loader(self.opts.train_dir)

        epoch_start = time.clock()
        epoch_train_cost = 0.0
        train_problems, train_filename = self.train_problems_loader.get_next()
        for problem in train_problems:
            samples = [sampler.sample(self.opts.list_size) for sampler in problem.sampler]
            candidates = [sample[0] for sample in samples]
            labels = [sample[1] for sample in samples]
            d = self.build_feed_dict(problem, candidates, labels)
            _, logits, cost = self.sess.run([self.apply_gradients, self.grades, self.cost], feed_dict=d)
            epoch_train_cost += cost

        epoch_train_cost /= len(train_problems)
        epoch_end = time.clock()

        learning_rate = self.sess.run(self.learning_rate)
        self.save(epoch)

        return (train_filename, epoch_train_cost, learning_rate, epoch_end - epoch_start)

    def test(self, test_data_dir):
        test_problems_loader = init_problems_loader(test_data_dir)
        results = []

        while test_problems_loader.has_next():
            test_problems, test_filename = test_problems_loader.get_next()

            epoch_test_cost = 0.0
            epoch_test_mat = ConfusionMatrix()

            for problem in test_problems:
                d = self.build_feed_dict(problem)
                logits, cost = self.sess.run([self.logits, self.cost], feed_dict=d)
                epoch_test_cost += cost
                epoch_test_mat.update(problem.is_sat, logits > 0)

            epoch_test_cost /= len(test_problems)
            epoch_test_mat = epoch_test_mat.get_percentages()

            results.append((test_filename, epoch_test_cost, epoch_test_mat))

        return results

class dummy_filter(object):
    def __init__(self):
        pass
    def filter(self, problem, trails, top_k):
        return trails

    # def find_solutions(self, problem):
    #     def flip_vlit(vlit):
    #         if vlit < problem.n_vars: return vlit + problem.n_vars
    #         else: return vlit - problem.n_vars

    #     n_batches = len(problem.is_sat)
    #     n_vars_per_batch = problem.n_vars // n_batches

    #     d = self.build_feed_dict(problem)
    #     all_votes, final_lits, logits, costs = self.sess.run([self.all_votes, self.final_lits, self.logits, self.predict_costs], feed_dict=d)

    #     solutions = []
    #     for batch in range(len(problem.is_sat)):
    #         decode_cheap_A = (lambda vlit: all_votes[vlit, 0] > all_votes[flip_vlit(vlit), 0])
    #         decode_cheap_B = (lambda vlit: not decode_cheap_A(vlit))

    #         def reify(phi):
    #             xs = list(zip([phi(vlit) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)],
    #                           [phi(flip_vlit(vlit)) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)]))
    #             def one_of(a, b): return (a and (not b)) or (b and (not a))
    #             assert(all([one_of(x[0], x[1]) for x in xs]))
    #             return [x[0] for x in xs]

    #         if self.solves(problem, batch, decode_cheap_A): solutions.append(reify(decode_cheap_A))
    #         elif self.solves(problem, batch, decode_cheap_B): solutions.append(reify(decode_cheap_B))
    #         else:

    #             L = np.reshape(final_lits, [2 * n_batches, n_vars_per_batch, self.opts.d])
    #             L = np.concatenate([L[batch, :, :], L[n_batches + batch, :, :]], axis=0)

    #             kmeans = KMeans(n_clusters=2, random_state=0).fit(L)
    #             distances = kmeans.transform(L)
    #             scores = distances * distances

    #             def proj_vlit_flit(vlit):
    #                 if vlit < problem.n_vars: return vlit - batch * n_vars_per_batch
    #                 else:                     return ((vlit - problem.n_vars) - batch * n_vars_per_batch) + n_vars_per_batch

    #             def decode_kmeans_A(vlit):
    #                 return scores[proj_vlit_flit(vlit), 0] + scores[proj_vlit_flit(flip_vlit(vlit)), 1] > \
    #                     scores[proj_vlit_flit(vlit), 1] + scores[proj_vlit_flit(flip_vlit(vlit)), 0]

    #             decode_kmeans_B = (lambda vlit: not decode_kmeans_A(vlit))

    #             if self.solves(problem, batch, decode_kmeans_A): solutions.append(reify(decode_kmeans_A))
    #             elif self.solves(problem, batch, decode_kmeans_B): solutions.append(reify(decode_kmeans_B))
    #             else: solutions.append(None)

    #     return solutions

    # def solves(self, problem, batch, phi):
    #     start_cell = sum(problem.n_cells_per_batch[0:batch])
    #     end_cell = start_cell + problem.n_cells_per_batch[batch]

    #     if start_cell == end_cell:
    #         # no clauses
    #         return 1.0

    #     current_clause = problem.L_unpack_indices[start_cell, 1]
    #     current_clause_satisfied = False

    #     for cell in range(start_cell, end_cell):
    #         next_clause = problem.L_unpack_indices[cell, 1]

    #         # the current clause is over, so we can tell if it was unsatisfied
    #         if next_clause != current_clause:
    #             if not current_clause_satisfied:
    #                 return False

    #             current_clause = next_clause
    #             current_clause_satisfied = False

    #         if not current_clause_satisfied:
    #             vlit = problem.L_unpack_indices[cell, 0]
    #             #print("[%d] %d" % (batch, vlit))
    #             if phi(vlit):
    #                 current_clause_satisfied = True

    #     # edge case: the very last clause has not been checked yet
    #     if not current_clause_satisfied: return False
    #     return True
