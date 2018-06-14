import scipy.io.wavfile  as wav
import matplotlib.pyplot as plt
import numpy             as np
import tensorflow        as tf
import shared_functions  as sf
import signal
import sys

conf_fn = sys.argv[1]
conf_fn = conf_fn[:-3]
print(conf_fn)

exec('import ' +  conf_fn + ' as conf')
config = conf.data

#===========global variables==========#

tm                = sf.Training_Manager()
'''
tm.lr             = 1e-4
tm.num_epochs     = 20
tm.seg_len        = 1024
tm.num_unrollings = 100
tm.num_layers     = 1
tm.int_cursor_inc = tm.seg_len
tm.ext_cursor_inc = 1
tm.weight_stddev  = 1e-6
tm.sampling_rate  = 11025
tm.label_shape    = (tm.num_unrollings, 1)
tm.label_offset   = tm.seg_len
tm.use_plotting   = False
tm.save_dreams    = True
tm.use_dropout    = True
tm.use_residual   = False
tm.regularizaton  = False
tm.audio_freq     = 10000
tm.comment        = ''
tm.input_fn_list = ['model_inputs/clarinet.wav']
'''


for _ in config:
    temp = 'tm.' + _ + ' = ' + str(config[_])
    exec(temp)
print('tm.lr', tm.lr)
print('tm.num_epochs', tm.num_epochs)
print('tm.seg_len', tm.seg_len)
print('tm.num_unrollings', tm.num_unrollings)
print('tm.num_layers', tm.num_layers)
print('tm.int_cursor_inc', tm.int_cursor_inc)
print('tm.ext_cursor_inc', tm.ext_cursor_inc)
print('tm.weight_stddev', tm.weight_stddev)
print('tm.sampling_rate', tm.sampling_rate)
print('tm.label_shape', tm.label_shape)
print('tm.label_offset', tm.label_offset)
print('tm.use_plotting', tm.use_plotting)
print('tm.save_dreams', tm.save_dreams)
print('tm.use_dropout', tm.use_dropout)
print('tm.use_residual', tm.use_residual)
print('tm.regularization', tm.regularization)
print('tm.audio_freq', tm.audio_freq)
print('tm.comment', tm.comment)
print('tm.input_fn_list', tm.input_fn_list)

reg_amount        = False
#dataset fns
noisify           = False
reset_dream_state = False
use_noise_state   = False
exec(sf.quit_gracefully) #this line injects quit_gracefully() from shared_functions
signal.signal(signal.SIGINT, quit_gracefully)

restore_from = ''

#read in all data
tm.read_in_data()
num_dream_samples = 11025*30





#===========model definition==========#
with tf.device(tm.device):
    #---placeholders---#
    model_input = tf.placeholder(tf.float32, [tm.num_unrollings, tm.seg_len], name='input_ph')
    label       = tf.placeholder(tf.float32, [tm.label_shape[0], tm.label_shape[1]], name='label_ph')

    #---LSTM initialization---#

    lstm_cell     = tf.contrib.rnn.LSTMCell(tm.seg_len)

    if tm.regularization != False:
        lstm_reg      = tf.contrib.layers.l2_regularizer(reg_amount, scope=None)
        lstm_cell.activity_regularizer = lstm_reg
    if tm.use_residual == True:
        lstm_cell     = tf.contrib.rnn.ResidualWrapper(lstm_cell)
    if tm.use_dropout  == True:
        lstm_cell     = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=0.5, output_keep_prob=1.0)

    lstm_cell     = tf.contrib.rnn.MultiRNNCell([lstm_cell]* tm.num_layers)
    cur_state     = lstm_cell.zero_state(tm.num_unrollings, tf.float32)
    state_ph      = tf.placeholder(tf.float32, [tm.num_layers, 2, tm.num_unrollings, tm.seg_len])
    state_unpack  = tf.unstack(state_ph, axis=0)
    state_tuple   = tuple([tf.contrib.rnn.LSTMStateTuple(state_unpack[i][0], state_unpack[i][1]) for i in range(tm.num_layers)])
    logits, state = lstm_cell(model_input, state_tuple)

    output       = tf.layers.dense(inputs=logits, units=1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=tm.weight_stddev))
    #layer         = tf.layers.dense(inputs=tf.transpose(layer_1), units=1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=tm.weight_stddev))



    #---loss and optimizer---#
    loss         = tf.losses.mean_squared_error(label, output)
    optim        = tf.train.AdamOptimizer(learning_rate=tm.lr).minimize(loss)


    # saver        = tf.train.Saver()


#===========run training==========#
sess = tf.Session()
if restore_from == '':
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, restore_from)

plt.ion()

plotter     = sf.Plotter()
loss_plot   = sf.Plottable(label='loss',       ls='-', color='black')
pred_plot   = sf.Plottable(label='prediction', ls='-', color='black')
targ_plot   = sf.Plottable(label='target',     ls='-', color='blue')

plots = {   0: [loss_plot],
            1: [pred_plot, targ_plot]
        }

losses      = []
mean_losses = []
val_losses  = []
test_losses = []
dream       = []


total_iters_completed = 0
dream_counter         = 0



cur_state       = np.zeros([tm.num_layers, 2, tm.num_unrollings, tm.seg_len])

if use_noise_state == True:
    cur_dream_state = np.random.normal(size=[tm.num_layers, 2, tm.num_unrollings, tm.seg_len])
else:
    cur_dream_state = np.zeros([tm.num_layers, 2, tm.num_unrollings, tm.seg_len])

seed, mode = tm.prep_seed()
ops_to_run = [loss, output, optim, state]
dream_ops  = [output, state]


for epoch in range(tm.num_epochs):

    print('\n' + '#' + '='*10 + str(epoch+1) + '/' + str(tm.num_epochs) + '='*10 + '#' + '\n')
    tm.train_cursor = 0
    for itr in range(tm.num_iters):
        total_iters_completed += 1

        #---prepare the inputs, labels---#
        cur_input, cur_label = tm.next(tm.input_data[0])
        if noisify != False:
            noise = np.random.normal(scale=noisify, size=(tm.num_unrollings, tm.seg_len))
            cur_input = cur_input + noise


        #---run the training---#
        cur_loss, cur_output, _, cur_state = sess.run(ops_to_run, feed_dict={
            model_input : cur_input,
            label       : cur_label,
            state_ph    : np.array(cur_state)})
        losses.append(cur_loss)



        #---do some printing---#
        if itr % tm.print_freq == 0:
            mean_loss = np.mean(losses[-tm.print_freq:])
            mean_losses.append(mean_loss)
            sf.print_loss(itr, tm.num_iters, ['mean loss'], [mean_loss])


        #---do some plotting---#
        if itr % tm.plot_freq == 0 and tm.use_plotting == True:
            loss_plot.item = mean_losses
            pred_plot.item = cur_output[0]
            targ_plot.item = cur_label[0]
            plotter.plot(plots)


        #---dream---#
        if tm.save_dreams == True:
            dream_counter += 1
            cur_dream_output, cur_dream_state = sess.run(dream_ops, feed_dict={
                model_input : seed,
                state_ph    : np.array(cur_dream_state)})
            seed         = np.roll(seed, -1)
            seed[-1][-1] = cur_dream_output[-1]
            dream.append(cur_dream_output[-1])
            if dream_counter == num_dream_samples:
                seed, mode = tm.prep_seed(force_mode='random')
                dream_fn = tm.base_name + tm.comment + '_' + mode + '_' + str(total_iters_completed) + '.wav'
                wav.write(dream_fn, tm.sampling_rate, np.hstack(np.array(dream)))
                print('saved ' + dream_fn)
                dream = []
                dream_counter = 0






loss_fn = 'outputs/loss.npy'
np.save(loss_fn, np.array(losses))
print('saved ' + loss_fn)
