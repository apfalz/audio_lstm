import scipy.io.wavfile  as wav
import matplotlib.pyplot as plt
import numpy             as np
import tensorflow        as tf
import shared_functions  as sf
import signal

#===========global variables==========#

tm                = sf.Training_Manager()

#training hyperparameters
tm.lr             = 1e-4
tm.num_epochs     = 20
tm.weight_stddev  = 1e-6 #1e-6 for vector#0.1 for transpose
tm.num_layers     = 1
tm.use_dropout    = True
tm.use_residual   = False
tm.regularization = True
tm.reg_amount     = 0.5
tm.input_dropout  = 0.5
tm.output_dropout = 1.0
tm.noisify        = False


#data shapes
tm.seg_len        = 1024
tm.num_unrollings = 100
tm.int_cursor_inc = 1
tm.ext_cursor_inc = 1
tm.label_shape    = (tm.num_unrollings, tm.seg_len)
tm.label_offset   = 1
tm.output_shape   = tm.label_shape
tm.use_transpose  = False


tm.sampling_rate  = 11025
#interface settings
tm.use_plotting   = False
tm.save_dreams    = True
tm.use_seq_dreams = True
tm.audio_freq     = tm.sampling_rate * 5
tm.print_freq     = 100
tm.plot_freq      = 1
tm.dream_length   = 400#tm.sampling_rate*5
tm.seq_dream_len  = 400#tm.sampling_rate*2
tm.base_name      = 'outputs/free_deb_11025fs_1024ws_256hs'
tm.comment        = ''
tm.verbose        = 2
#dataset fns
tm.input_fn_list  = ['model_inputs/free_deb.wav']


#fft hyperparameters
tm.use_fft        = True
tm.hop_size       = 256
tm.window_size    = tm.seg_len
tm.zero_padding   = True


restore_from = ''

#read in all data
tm.read_in_data()


exec(sf.quit_gracefully) #this line injects quit_gracefully() from shared_functions
signal.signal(signal.SIGINT, quit_gracefully)

tm.save_json()


#===========model definition==========#
with tf.device(tm.device):
    #---placeholders---#
    model_input = tf.placeholder(tf.float32, [tm.num_unrollings, tm.seg_len],        name='input_ph')
    label       = tf.placeholder(tf.float32, [tm.label_shape[0], tm.label_shape[1]], name='label_ph')

    #---LSTM initialization---#

    lstm_cell     = tf.contrib.rnn.LSTMCell(tm.seg_len)

    if tm.regularization != False:
        lstm_reg      = tf.contrib.layers.l2_regularizer(tm.reg_amount, scope=None)
        lstm_cell.activity_regularizer = lstm_reg
    if tm.use_residual == True:
        lstm_cell     = tf.contrib.rnn.ResidualWrapper(lstm_cell)
    if tm.use_dropout  == True:
        lstm_cell     = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=tm.input_dropout, output_keep_prob=tm.output_dropout)

    lstm_cell     = tf.contrib.rnn.MultiRNNCell([lstm_cell]* tm.num_layers)
    cur_state     = lstm_cell.zero_state(tm.num_unrollings, tf.float32)
    state_ph      = tf.placeholder(tf.float32, [tm.num_layers, 2, tm.num_unrollings, tm.seg_len])
    state_unpack  = tf.unstack(state_ph, axis=0)
    state_tuple   = tuple([tf.contrib.rnn.LSTMStateTuple(state_unpack[i][0], state_unpack[i][1]) for i in range(tm.num_layers)])
    logits, state = lstm_cell(model_input, state_tuple)

    if tm.use_transpose == True:
        layer     = tf.layers.dense(inputs=logits, units=1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=tm.weight_stddev))
        output    = tf.layers.dense(inputs=tf.transpose(layer), units=1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=tm.weight_stddev))
    else:
        output       = tf.layers.dense(inputs=logits, units=tm.output_shape[1], kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=tm.weight_stddev))


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
        if tm.noisify != False:
            noise = np.random.normal(scale=tm.noisify, size=(tm.num_unrollings, tm.seg_len))
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

        #---sequential_dream---#
        if tm.save_dreams == True and tm.use_seq_dreams == True and total_iters_completed % tm.audio_freq == 0:
            prog                = sf.Progress_Bar(tm.seq_dream_len)
            seq_dream           = []
            seq_dream_counter   = 0
            seq_dream_state     = np.zeros([tm.num_layers, 2, tm.num_unrollings, tm.seg_len])
            seq_dream_seed,mode = tm.prep_seed()
            seq_dream_ops       = [output, state]
            for d in range(tm.seq_dream_len):
                cur_seq_dream_output, cur_seq_dream_state = sess.run(seq_dream_ops, feed_dict={
                    model_input : seq_dream_seed,
                    state_ph    : np.array(seq_dream_state)})
                if tm.use_fft == False and tm.label_shape[1] == 1:
                    seq_dream_seed         = np.roll(seq_dream_seed, -1)
                    seq_dream_seed[-1][-1] = cur_seq_dream_output[-1]
                elif tm.use_fft == False and tm.label_shape[1] != 1:
                    seq_dream_seed         = seq_dream_seed[1:]
                    seq_dream_seed         = np.append(seq_dream_seed, cur_seq_dream_output[-1].reshape((1, -1)), axis=0)
                else:
                    seq_dream_seed         = seq_dream_seed[1:]
                    seq_dream_seed         = np.append(seq_dream_seed, cur_seq_dream_output[-1].reshape((1, -1)), axis=0)
                seq_dream.append(cur_seq_dream_output[-1])
                prog.update(d)
            if tm.use_fft == False:
                seq_dream_fn = tm.base_name + 'sequential_dream' + tm.comment + '_' + str(total_iters_completed) + '.wav'
                wav.write(seq_dream_fn, tm.sampling_rate, np.hstack(np.array(seq_dream)))
                print('saved ' + seq_dream_fn)

        #---concurrent_dream---#
        if tm.save_dreams == True:
            dream_counter += 1
            cur_dream_output, cur_dream_state = sess.run(dream_ops, feed_dict={
                model_input : seed,
                state_ph    : np.array(cur_dream_state)})
            if tm.use_fft == False and tm.label_shape[1] == 1:
                seed         = np.roll(seed, -1)
                seed[-1][-1] = cur_dream_output[-1]
            elif tm.use_fft == False and tm.label_shape[1] != 1:
                seed         = seed[1:]
                seed         = np.append(seed, cur_dream_output[-1].reshape((1, -1)), axis=0)
            else:
                seed         = seed[1:]
                seed         = np.append(seed, cur_dream_output[-1].reshape((1, -1)), axis=0)

            dream.append(cur_dream_output[-1])
            if dream_counter >= tm.dream_length:
                seed, mode = tm.prep_seed()
                dream_fn = tm.base_name + tm.comment + '_' + mode + '_' + str(total_iters_completed) + '.wav'
                if tm.use_fft == False:
                    wav.write(dream_fn, tm.sampling_rate, np.hstack(np.array(dream)))
                else:
                    dream = sf.to_audio_new(dream, input_hop_size=tm.hop_size, window_size=tm.window_size, sampling_rate=tm.sampling_rate)
                    wav.write(dream_fn, tm.sampling_rate, dream)

                print('saved ' + dream_fn)
                dream = []
                dream_counter = 0






loss_fn = 'outputs/' + tm.base_name + 'loss.npy'
np.save(loss_fn, np.array(losses))
print('saved ' + loss_fn)
