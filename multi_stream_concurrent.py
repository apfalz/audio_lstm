import scipy.io.wavfile  as wav
import matplotlib.pyplot as plt
import numpy             as np
import tensorflow        as tf
import shared_functions  as sf
import signal
import sys

sys.path.append('./presets')

conf_fn = sys.argv[1]
conf_fn = conf_fn[:-3]
print(conf_fn)

exec('import ' +  conf_fn + ' as conf')
config = conf.data

num_dream_streams = 3

#===========global variables==========#

tm                = sf.Training_Manager()


#set all hyperparameters accoriding to config dict
for _ in config:
    if type(config[_]) == str:
        temp = 'tm.' + _ + ' = ' + repr(config[_])
    else:
        temp = 'tm.' + _ + ' = ' + str(config[_])
    print (temp)
    exec(temp)

#print all hyperparameters
for _ in tm.__dict__:
    print(_ + ' : ' + str(tm.__dict__[_]))

def test_dream(fs=11025):
    wav.write('test.wav', fs, np.hstack(np.array(dream)))
    print('saved dream as test.wav')

def test(fs=11025):
    for i in range(num_dream_streams):
        fn = 'test' + '_' + str(i) + '.wav'
        wav.write(fn, fs, np.hstack(np.array(dreams[i])))
        print('saved ' + fn)




def quit_gracefully(signum, frame):
    print('\\n#' + '~'*8 + 'process halted' + '~'*8 + '#')
    done = False
    while(done==False):
        try:
            code = input("Code to execute: \n>> ")
            if code != '':
                exec(code, globals())
                done = True
        except:
            print('Previous code injection failed.')
        end = input("Keep going? (True/False) \n>> ")
        if end != '':
            if end == 'False' or end == '0':
                print('trying to quit...')
                #TODO form guaranteed unique output filename. Save hyperparameters and possibly loss curves, other data.
                raise SystemExit

signal.signal(signal.SIGINT,  quit_gracefully)
signal.signal(signal.SIGTERM, quit_gracefully)

#read in all data
tm.read_in_data()





#===========model definition==========#
with tf.device(tm.device):
    #---placeholders---#
    model_input = tf.placeholder(tf.float32, [tm.num_unrollings, tm.seg_len], name='input_ph')
    label       = tf.placeholder(tf.float32, [tm.label_shape[0], tm.label_shape[1]], name='label_ph')

    #---LSTM initialization---#

    lstms         = []
    for i in range(tm.num_layers):
        cell = tf.contrib.rnn.LSTMCell(tm.seg_len)
        if tm.use_dropout  == True:
            cell     = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=tm.input_dropout, output_keep_prob=tm.output_dropout, state_keep_prob=tm.state_keep_prob)
        if tm.use_residual == True:
            cell     = tf.contrib.rnn.ResidualWrapper(cell)
        lstms.append(cell)
    #
    #
    # if tm.regularization != False:
    #     lstm_reg      = tf.contrib.layers.l2_regularizer(tm.reg_amount, scope=None)
    #     lstm_cell.activity_regularizer = lstm_reg


    lstm_cell     = tf.contrib.rnn.MultiRNNCell(lstms)
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
sess.run(tf.global_variables_initializer())
#if restore_from == '':
#    sess.run(tf.global_variables_initializer())
#else:
#saver.restore(sess, restore_from)

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


total_iters_completed = 0
dream_counter         = 0



cur_state       = np.zeros([tm.num_layers, 2, tm.num_unrollings, tm.seg_len])


dream_states = []
seeds        = []
dreams       = []
for i in range(num_dream_streams):
    dream_states.append(np.zeros([tm.num_layers, 2, tm.num_unrollings, tm.seg_len]))
    seed, mode = tm.prep_seed()
    seeds.append(seed)
    dreams.append([])



ops_to_run = [loss, output, optim, state]
dream_ops  = [output, state]




for epoch in range(tm.num_epochs):

    print('\n' + '#' + '='*10 + str(epoch+1) + '/' + str(tm.num_epochs) + '='*10 + '#' + '\n')
    tm.train_cursor = 0
    for itr in range(tm.num_iters):
        total_iters_completed += 1

        if tm.reset_state != False and total_iters_completed % tm.reset_state == 0:
            cur_state       = np.zeros([tm.num_layers, 2, tm.num_unrollings, tm.seg_len])


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
            for current in range(num_dream_streams):


                cur_dream_output, dream_states[current] = sess.run(dream_ops, feed_dict={
                    model_input : seeds[current],
                    state_ph    : np.array(dream_states[current])})
                if tm.use_fft == False and tm.label_shape[1] == 1:
                    seeds[current]         = np.roll(seeds[current], -1)
                    seeds[current][-1][-1] = cur_dream_output[-1]
                elif tm.use_fft == False and tm.label_shape[1] != 1:
                    #todo handle non-samplewise prediction
                    seed         = seed[1:]
                    seed         = np.append(seed, cur_dream_output[-1].reshape((1, -1)), axis=0)
                else:
                    #todo handle fft? maybe don't need this?
                    seed         = seed[1:]
                    seed         = np.append(seed, cur_dream_output[-1].reshape((1, -1)), axis=0)

                dreams[current].append(cur_dream_output[-1])
            if dream_counter >= tm.dream_length:
                dreams = []
                for d in range(num_dream_streams):
                    #reset all states
                    dream_states[d] = np.zeros([tm.num_layers, 2, tm.num_unrollings, tm.seg_len])
                    #generate all new seeds
                    seed, mode = tm.prep_seed()
                    seeds[d] = seed
                    #save generated dreams
                    dream_fn = tm.base_name + tm.comment + '_' + str(total_iters_completed) + '_' + str(d) + '.wav'
                    if tm.use_fft == False:
                        wav.write(dream_fn, tm.sampling_rate, np.hstack(np.array(dreams[d])))
                    else:
                        dream = sf.to_audio_new(dreams[d], input_hop_size=tm.hop_size, window_size=tm.window_size, sampling_rate=tm.sampling_rate)
                        wav.write(dream_fn, tm.sampling_rate, dreams[d])
                    print('saved ' + dream_fn)
                    #empty dreams list element
                    dreams[d] = []
                dream_counter = 0






loss_fn = 'outputs/' + tm.base_name + 'loss.npy'
np.save(loss_fn, np.array(losses))
print('saved ' + loss_fn)
