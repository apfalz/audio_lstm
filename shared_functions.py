from __future__ import print_function
import sys
import numpy             as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wav
import tensorflow        as tf
import signal
import time
#======================================================================#


quit_gracefully = """
def quit_gracefully(signum, frame):
    print('\\n#' + '~'*8 + 'process halted' + '~'*8 + '#')
    try:
        code = raw_input("Code to execute: \\n>> ")
        if code != '':
            exec(code, globals())
    except:
        print('Previous code injection failed.')
        code = raw_input("Code to execute: \\n>> ")
        if code != '':
             exec(code in globals(), globals())
    end = input("Keep going? (True/False) \\n>> ")
    if end != '':
        if end == False:
            raise SystemExit"""


def time_it(func):
    '''
    timing function for use as decorator:
    #-----------------------#
    @time_it
    def example_func(argu):
        return arg
    #-----------------------#
    >>example_func('testing decorator')
    example_func took 3.099e-6 to execute.
    'testing decorator'

    '''
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        end    = time.time()
        print(func.__name__ + " took " + str(end - start) + ' seconds to execute.')
        return result
    return wrapper

def print_loss(cur_itr, num_iters, name_list, loss_list, comment=None):
    fill_amount = len(str(num_iters))
    #loss_list   = ['{:.5}'.format(i).zfill(8) for i in loss_list]
    loss_list   = ['{:.5}'.format(i) for i in loss_list]
    string      = str(cur_itr).zfill(fill_amount) + '/' + str(num_iters)
    for i in range(len(loss_list)):
        string = string + ' ' + name_list[i] + ': '
        string = string + ' ' + loss_list[i]
    if comment != None:
        string = string + ' ' + comment

    print(string)

def mtof(pitch):
    pitch = float(pitch)
    return 440.0 * 2.0**((pitch - 69.0)/12.0)

def adsr(length, num_segments=4):
    points = [0.0]
    output = []


    '''generate heights for internal points'''
    for i in range(num_segments - 1):
        points.append((np.random.random()*0.5) + 0.5)

    '''generate places for the internal points'''
    places = gen_randomly_spaced_indices(length, num_segments)

    '''generate adsr'''
    for i in range((len(points) - 1)):
        output.append(np.linspace(points[i], points[i+1], places[i+1] - places[i]))
    output.append(np.linspace(points[-1], 0.0, length - places[-1]))

    output = np.array(output)
    output = np.hstack(output)
    return output

def gen_randomly_spaced_indices(desired_length, num_segments):
    max_shift = (desired_length // num_segments ) // 4
    shift_min = -(max_shift // 4)
    shift_max =  (max_shift // 4)
    even_segs = desired_length  // num_segments
    indices   = range(desired_length)
    output    = [0]
    for i in range(1, num_segments):
        output.append(even_segs*i)

    for i in range(1, len(output)):
        output[i] = output[i]# + np.random.randint(shift_min, shift_max)
    return sorted(output)

def random_step_function(min_midi, max_midi, output_length_in_samples, same_length_steps=False, min_step_length=1000, max_step_length=10000, normalize=False):
    '''returns a random squarewave of frequencies '''
    output = []
    total  = 0
    while total < output_length_in_samples:

        if same_length_steps != False:
            step_len = same_length_steps
        else:
            step_len =      np.random.randint(min_step_length, max_step_length)

        total       +=      step_len
        cur_freq     = mtof(np.random.randint(min_midi,     max_midi    ))
        temp         = np.ones(step_len) * cur_freq
        output.append(temp)
    output  = np.hstack(np.array(output))
    if len(output) != output_length_in_samples:
        output = output[:output_length_in_samples]

    if normalize==True:
        output = output / np.amax(output)

    return output

def to_one_hot(data, num_classes, scale=None):

    zeros = np.zeros(len(data))
    zeros[data] = 1.0
    return zeros

def unwrap(data):
    copy    = np.array(data, copy=True)
    for i in range(len(copy) - 1):
        if copy[i+1] < copy[i]:
            copy[i+1:] = copy[i+1:] + np.pi*2.0
    return copy

def interpolate(data, hop_size):
    output = []
    for i in range(len(data) - 1):
        print(output)
        output.append(np.linspace(data[i], data[i+1], hop_size, endpoint=False))
    return np.array(output).reshape((1, -1))

def poltocar(mag, phase):
    car = []
    for i in range(len(mag)):
        car.append(complex(mag[i]*np.cos(phase[i]), np.sin(phase[i])))
    return car

#===========================FM functions========================#
def tf_osc(freq, fs=44100):
    phases = tf.cumsum(freq)
    phases = phases  / float(fs) * np.pi * 2.0
    return tf.sin(phases)

def fm_synth(cf, hr, mi, am, fs):
    mf = cf * hr
    md = mf * mi
    return am * tf_osc(cf + md* tf_osc(mf, fs=fs),fs=fs)

def osc(freq, sampling_rate=44100):
    phases = np.cumsum(freq)
    phases = (phases % (sampling_rate - 1)) / float(sampling_rate) * np.pi * 2.0
    return np.sin(phases)

def osc_cos(freq, sampling_rate=44100):
    phases = np.cumsum(freq)
    phases = (phases % (sampling_rate - 1)) / float(sampling_rate) * np.pi * 2.0
    return np.cos(phases)


def fm_no_amp(cf, hr, mi, sampling_rate):
    mf = cf * hr
    md = mf * mi
    return osc(cf + md*osc(mf, sampling_rate=sampling_rate), sampling_rate=sampling_rate)

def fm(cf, hr, mi, amplitude, sampling_rate):
    mf = cf * hr
    md = mf * mi
    return amplitude * osc(cf + md*osc(mf, sampling_rate=sampling_rate), sampling_rate=sampling_rate)

def simple_fm(cf, mf, ms, am):
    return osc( (osc(mf, sampling_rate=sampling_rate)*ms)  + cf , sampling_rate=sampling_rate)* am

def to_real(data):
    '''takes in audio, returns normalized magnitudes'''
    output   = []
    half_len = len(data[0]) / 2
    for i in data:
        temp = np.fft.fft(i * np.blackman(len(i)))[:half_len]
        output.append(np.sqrt(np.square(temp.real) + np.square(temp.imag)))
    output = np.array(output)
    return output / np.amax(output)

def to_real_better(data):
    '''takes in chunked audio. returns magnitude spectra with zero padding.'''
    output = []
    for i in data:
        temp = i * np.blackman(len(i))
        temp = np.hstack(np.array([temp, np.zeros(len(i))]))
        temp = np.fft.fft(temp)
        temp = np.sqrt(np.square(temp.real) + np.square(temp.imag))[:len(i)]
        output.append(temp)
    output = np.array(output)
    return output / np.amax(output)

def to_real_keep_phase(data, padded=False):
    '''takes in chunked audio returns normalized fft spectrum with phase information'''
    output = []
    size   = len(data[0])
    window = np.blackman(size)

    for i in range(len(data)):
        if padded == True:
            spec_input = np.hstack(np.array([data[i]*window, [0.0]*size]))
            spec = np.fft.fft(spec_input)
            spec = spec[:size  ]
        else:
            spec = np.fft.fft(data[i] * window)
            spec = spec[:size/2]
        temp = []
        for j in spec:
            temp.append(j.real)
            temp.append(j.imag)
        output.append(temp)
    output = np.array(output)
    output = output / np.amax(output)
    return output

#============================resynthesis functions=========================#
def get_bins(len_data, sampling_rate=11025):
    '''N (Bins) = FFT Size/2
             FR = Fmax/N(Bins)'''
    fft_size = len_data
    f_max    = sampling_rate / 2.0
    bin_size = f_max / (fft_size)
    bin_size = f_max / (fft_size /2.0)
    freqs    = [i * bin_size for i in range(fft_size)]
    freqs    = [i * bin_size for i in range(fft_size / 2)]

    return freqs

def get_bins_new(len_data, sampling_rate=11025):
    '''N (Bins) = FFT Size/2
             FR = Fmax/N(Bins)'''
    fft_size = len_data
    f_max    = sampling_rate / 2.0
    #f_max    = sampling_rate
    bin_size = f_max / (fft_size)

    freqs    = [i * bin_size for i in range(fft_size)]

    return freqs

def get_peaks(data):
    '''expects a two dimensions'''
    peaks = []
    for j in range(len(data)):
        temp        = []
        num_omitted = 0
        temp.append(0.)
        for i in range(1, len(data[j])-1, 1):
            if data[j][i] < 0.0:
                data[j][i] = 0.0
            if (data[j][i] > data[j][i-1] and data[j][i] > data[j][i+1]):
                temp.append(data[j][i])
            else:
                num_omitted += 1
                temp.append(0.0)
        temp.append(0.)
        peaks.append(temp)
    print('number of bins omitted', num_omitted)
    return peaks

def make_envelope(data, hop_size=2048):
    env = []
    for i in range(len(data) - 1):
        if i == 0:
            env.append(np.linspace(0.,       data[0],   hop_size))
        env.append(    np.linspace(data[i],  data[i+1], hop_size))
    env.append(        np.linspace(data[-1], 0.,        hop_size))
    env = np.array(env).flatten()
    return env

def to_audio(data, threshold = 500, input_hop_size=2048, window_size=2048, sampling_rate = 11025):
    '''
    read in the data
    tranpose
    create smoothed amplitude signal
    generate sine and multiply by amplitude element-wise
    add result to output
    normalize the output
    '''

    freqs           = get_bins(window_size, sampling_rate = sampling_rate)
    max_term        = np.amax(data)
    norm_term       = 1.0 / np.amax(data)
    threshold       = max_term - threshold #1000 will give more frequencies, 100 will give less frequencies

    peaks           = get_peaks(data)
    rows            = np.transpose(peaks)
    dummy_envelope  = make_envelope(rows[0], hop_size=input_hop_size)
    output          = np.zeros(len(dummy_envelope))
    num_freqs       = 0
    two_pi          = 2.0*np.pi
    samp            = 1.0/sampling_rate
    len_envelope    = len(dummy_envelope)
    len_rows        = len(rows)
    offset          = 0.0
    for i in xrange(len_rows):
        offset     = np.random.randint(200)
        counter    = np.arange(offset, len_envelope + offset)
        envelope   = make_envelope(rows[i],hop_size=input_hop_size)
        num_freqs += 1

        update     =  two_pi * freqs[i] * samp
        the_range  = np.sin(counter * update) * envelope
        output     = output + the_range

    output /= np.amax(output)

    return output

def to_audio_new(data, threshold = 500, input_hop_size=2048, window_size=2048, sampling_rate = 11025):
    '''
    read in the data
    tranpose
    create smoothed amplitude signal
    generate sine and multiply by amplitude element-wise
    add result to output
    normalize the output
    '''

    freqs           = get_bins_new(window_size, sampling_rate = sampling_rate)
    max_term        = np.amax(data)
    norm_term       = 1.0 / np.amax(data)
    threshold       = max_term - threshold #1000 will give more frequencies, 100 will give less frequencies

    peaks           = get_peaks(data)
    rows            = np.transpose(peaks)
    dummy_envelope  = make_envelope(rows[0], hop_size=input_hop_size)
    output          = np.zeros(len(dummy_envelope))
    num_freqs       = 0
    two_pi          = 2.0*np.pi
    samp            = 1.0/sampling_rate
    len_envelope    = len(dummy_envelope)
    len_rows        = len(rows)
    offset          = 0.0
    prog            = Progress_Bar(len_rows)
    for i in xrange(len_rows):
        prog.update(i)
        offset     = np.random.randint(200)
        counter    = np.arange(offset, len_envelope + offset)
        envelope   = make_envelope(rows[i],hop_size=input_hop_size)
        num_freqs += 1
        update     =  two_pi * freqs[i] * samp
        the_range  = np.sin(counter * update) * envelope
        output     = output + the_range

    output /= np.amax(output)

    return output

def keep_only(data, num):
    '''take in magnitude spectra, return windows where only the num loudest bins remain'''
    output  = []
    copy    = data[:]
    seg_len = len(copy[0])
    for window in copy:
        temp    = np.zeros(seg_len)
        for bin in range(num):
            #find the bin with the most power. Set that bin in temp, then zero the bin in window.
            max_bin         = np.argmax(window)

            temp[max_bin]   = window[max_bin]
            window[max_bin] = 0.0
        output.append(temp)

    return np.array(output)


#============================generic functions=========================#
def upsample(data, amount):
    '''Simple upsampling algorithm. Just repeat each value in data amount times. Assumes a single array for data.'''
    output = []
    for i in data:
        output.append([i]*amount)
    return np.hstack(np.array(output))

def how_many_matrices(chunked_data, num_unrollings, cursor_inc):
    num    = 0
    cursor = 0
    while len(chunked_data[cursor:]) >= num_unrollings:
        cursor += cursor_inc
        num    += 1
    return num

def read_in_data(input_fn_list, chunk=True, seg_len=1024, cursor_inc=1024, truncate_amount=False, pad_data=False):

    sampling_rates = []
    output         = []

    for i in range(len(input_fn_list)):
        if input_fn_list[i] != '':
            if input_fn_list[i][-3:] == 'wav':
                fs, cur_output = wav.read(input_fn_list[i])
                sampling_rates.append(fs)
                if pad_data != False:
                    padding = np.zeros(pad_data)#pass in number of zeros to pad with instead of False
                    cur_output = np.hstack(np.array([padding, cur_output]))
                if chunk == True:
                    cur_output = break_into_chunks(cur_output, seg_len, cursor_inc, truncate=truncate_amount)
            elif input_fn_list[i][-3:] == 'npy':
                cur_output = np.load(input_fn_list[i])
            output.append(cur_output)
        else:
            #this allows for output list to always match the number of inputs.
            output.append('')
        if len(sampling_rates) > 1:
            for i in range(1, len(sampling_rates)):
                if sampling_rates[i] != sampling_rates[0]:
                    print('found mismatched sampling rates!')
                    print(sampling_rates)


    return output

def break_into_chunks(data, size, overlap_amount, truncate=False):
    if truncate != False:
        data = data[:truncate]
    output = []
    cursor = 0
    while len(data[cursor:]) >= size:
        output.append(data[cursor:cursor+size])
        cursor += overlap_amount
    return np.array(output)

def column_reduce(data, seg_len, use_mean=False):
    output = []
    data   = break_into_chunks(data, seg_len, seg_len)
    mid    = seg_len / 2
    for d in data:
        if use_mean == True:
            output.append(np.mean(d))
        else:
            output.append(d[mid])
    return np.array(output)

def break_into_chunks_2D_to_3D(data, size, overlap_amount):
    '''break 2d array in to 3d chunks'''
    output = []
    cursor = 0
    while len(data[0][cursor:]) >= size:
        output.append(data[:, cursor:cursor+size])
        cursor += overlap_amount
    return np.array(output)

def smooth(data, first=0.5, last=0.5):
    prev_sum = data[0]
    output   = []
    for i in range(len(data)):
        cur_sum  = (data[i]*first) + (prev_sum*last)
        prev_sum = cur_sum
        output.append(cur_sum)
    return np.array(output)

def sliding_average(data, window_size=32):
    first_padding = np.ones(window_size/2) * data[ 0]
    last_padding  = np.ones(window_size/2) * data[-1]
    padded        = np.array([first_padding, data, last_padding])
    padded        = np.hstack(padded)
    output        = []
    for i in range(len(data)):
        output.append(np.mean(data[i:i+window_size]))
    return np.array(output)

def gen_random_envelope(output_len_in_samples, seg_min, seg_max):
	output        = []
	cur_point     = 0.0
	prev_point    = np.random.random()
	total_samples = 0
	len_seg       = np.random.randint(seg_min, seg_max)
	while total_samples + len_seg  < output_len_in_samples:
		cur_point      = np.random.random()
		total_samples += len_seg
		seg            = np.linspace(prev_point, cur_point, len_seg)

		len_seg        = np.random.randint(seg_min, seg_max)
		prev_point     = cur_point
		output.append(seg)

	final_seg = np.linspace(prev_point, np.random.random(), output_len_in_samples - total_samples)
	output.append(final_seg)
	output = np.array(output)
	output = np.hstack(output)
	if len(output) > output_len_in_samples:
		print('oops! envelope generated by gen_random_envelope() is too long.... truncating')
		output = output[:output_len_in_samples]

	return output

def interpolate_control_signals(data,  seg_len=1024, interpolate=True):
    '''expects a [num_unrollings x 4] array of data'''
    data   = data.T
    output = []
    if interpolate == True:
        for i in range(4):
            temp = []
            for j in range(len(data[i]) -1):
                temp.append(np.linspace(data[i][j], data[i][j+1], seg_len, endpoint=False))
            temp = np.hstack(np.array(temp))
            output.append(temp)
    else:
        pass

    return output


#============================swell functions for samplewise lstm model=========================#
def gen_swells(output_len_in_samples, num_swells, swell_width, verbose=False, max_val=1.0):
	#generate ramps, and locations
	output        = np.zeros(output_len_in_samples)
	separation    = output_len_in_samples // (num_swells + 1)
	swell_centers = [separation*(i+1) for i in range(num_swells)]
	if len(swell_centers) != num_swells:
		print('wrong number of swells')
	ramp_length   = swell_width // 2
	ramp_in       = np.linspace(0.,      max_val, ramp_length)
	ramp_out      = np.linspace(max_val, 0.,      ramp_length)
	swell         = np.hstack((ramp_in, ramp_out))
	swell_starts  = []

	for i in range(len(swell_centers)):
		try:
			swell_starts.append(swell_centers[i] - ramp_length)
		except:
			print('not able to define swell starts')
	if verbose == True:
            print('\n' + 'swell_starts' + str(swell_starts) + '\n')
	#apply the ramps
	for i in range(len(swell_starts)):
		try:
            #if verbose == True:
            #    where_to_put = output[swell_starts[i]: swell_starts[i] + swell_width]
            #    print('len(swell):' + str(len(swell)))
            #    print('applied to:' + str(len(where_to_put)))
            #    print('len(output)' + str(len(output)))

			output[swell_starts[i]:swell_starts[i]+len(swell)] = swell
		except:
			print('i:' + str(i))
			print('not able to apply swells')
			num_swells -= 1
			return gen_swells(output_len_in_samples, num_swells, swell_width)
	if verbose == True:
            print('len(output): ' + str(len(output)))
            print('swell_starts:' + str(swell_starts))

            plt.ioff()
            plt.plot(output, label='output envelope')
            plt.show()
            quit()

	return output

def gen_inverse_envelope(envelope):
	envelope = np.array(envelope)
	return 1.0 - envelope

def fade_in_and_out(output_length_in_samples, max_val=1.0):
    output = []
    third  = output_length_in_samples // 3
    output.append(np.linspace(max_val, 0.0,     third))
    output.append(np.zeros(third))
    output.append(np.linspace(0.0,     max_val, third))
    output = np.hstack(np.array(output))
    print(output.shape)
    if len(output) > output_length_in_samples:
        output = output[:output_length_in_samples]
    while len(output) < output_length_in_samples:
        output = np.append(output, 1.0)
    return output

def trapezoid(output_length_in_samples, ramp_length, max_val=1.0, verbose=False):
    ramp_in  = np.linspace(max_val, 0.0,    ramp_length)
    ramp_out = np.linspace(0.0,     max_val,ramp_length)
    zeros    = np.zeros(output_length_in_samples - ramp_length*2)
    output   = np.array([ramp_in, zeros, ramp_out])
    output   = np.hstack(output)
    if verbose == True:
        plt.ioff()
        plt.plot(output)
        plt.show()
    return output

#================================samp2samp functions==========================#
def get_maxes(data, num):
    '''takes in normalized magnitudes, returns frames with only loud bins and zeros for rest'''
    output = []
    for i in range(len(data)):
        maxes  = []
        temp = np.array(data[i], copy=True)
        for j in range(num):
            cur_max = np.argmax(temp)
            maxes.append(cur_max)
            temp[cur_max] = 0.0
        mask = np.zeros(len(data[0]))
        for k in range(num):
            mask[maxes[k]] = 1.0
        output.append(data[i] * mask)
    return np.array(output)
#================================class definitions============================#
class Plottable:
    def __init__(self, label='', ls='-', color='black', item=None, ylim=None, title=''):
        self.label = label
        self.ls    = ls
        self.color = color
        self.item  = item
        self.ylim  = ylim
        self.title = title


class Plotter:

    colors  = [None, 'black', 'blue', 'red', 'fuchsia', 'gold', 'green', 'purple', 'teal', 'yellow']
    lines   = ['-', '--', '-.', ':']
    counter = 0


    def plot(self, plot_dict):
        '''expects a dictionary that maps counter values to lists of instances of Plottable '''
        cur_list = plot_dict[self.counter]
        plt.pause(0.0001)
        plt.clf()
        for plottable in cur_list:

            self.sanitize(plottable)

            plt.plot(plottable.item, label=plottable.label, color=plottable.color, ls=plottable.ls)
        plt.legend()
        plt.title(cur_list[0].title)
        plt.ylim(plottable.ylim)
        plt.show()
        self.counter += 1



        self.counter %= len(plot_dict)

    def sanitize(self, plottable):
        '''expects an instance of Plottable'''
        if plottable.ls not in self.lines:
            plottable.ls = '-'
        if plottable.color not in self.colors:
            plottable.color = 'black'


class Progress_Bar:
    total_progress = 0

    def __init__(self, total_iters, percentage=10, hash_per_prog=2):
        self.total_progress = 0
        self.total_iters    = total_iters
        self.percentage     = percentage
        self.increment      = total_iters / percentage
        self.hash_per_prog  = hash_per_prog
        self.string         = '|' + '-'*(self.percentage*hash_per_prog) + '|'

    def update(self, current_index):
        '''check to see if we need to add some progress, if so print'''
        if current_index % self.increment == 0 and current_index != 0:
            self.total_progress += 1
            num_hashes           = self.total_progress*self.hash_per_prog
            num_dashes           = self.hash_per_prog*self.percentage - num_hashes
            self.string = '|' + '#'*num_hashes + '-'*num_dashes + '|'
            print(self.string, end='\r')
            sys.stdout.flush()

        if current_index == self.total_iters -1:
            print('|' + '='*(self.percentage*self.hash_per_prog) + '| Done!', end='\r')
            sys.stdout.flush()
            self.string = '|' + '-'*(self.percentage*self.hash_per_prog) + '|'
            self.total_progress = 0

    def cleanup(self):
        print(' ' * ((self.percentage*self.hash_per_prog) + 8), end='\r')
        sys.stdout.flush()


class Urn:
    def __init__(self, count):
        self.count      = count
        self.randomized = np.arange(count)
        np.random.shuffle(self.randomized)
        self.cur_index  = -1 #make sure all indices are used

    def next(self):
        self.cur_index += 1
        return self.randomized[self.cur_index % self.count]

    def reshuffle(self):
        self.randomized = np.arange(self.count)
        np.random.shuffle(self.randomized)


class Shuffled_Segments:
    '''generate flattened index array that is split into  segments, shuffled. Also remember indices where lstm state should be reset'''
    def __init__(self, total_iters, num_segments, verbose):
        self.total_iters  = total_iters
        self.num_segments = num_segments
        self.verbose      = verbose
        self.counter      = 0

        self.seg_length   = total_iters / num_segments
        self.normal       = np.arange(total_iters)
        self.chunked      = []
        cursor            = 0
        for i in range(num_segments - 1):
            self.chunked.append(self.normal[cursor:cursor+self.seg_length])
            cursor += self.seg_length
        self.chunked.append(self.normal[cursor:])
        shuffled_indices   = np.arange(len(self.chunked))
        np.random.shuffle(shuffled_indices)
        self.shuffled      = np.array([self.chunked[i] for i in shuffled_indices])
        lengths            = np.array([len(i) for i in self.shuffled])
        self.reset_indices = np.cumsum(lengths)[:-1]
        self.reset_indices = np.insert(self.reset_indices, 0, 0)
        self.flattened     = np.hstack(self.shuffled)

    def next(self):
        if self.counter in self.reset_indices:
            reset = True
            if self.verbose  == True:
                print('-'*5 + 'resetting state' + '-'*5)
        else:
            reset = False

        return_tuple  = (self.flattened[self.counter], reset)
        self.counter += 1
        self.counter %= self.total_iters
        return return_tuple


class Stream_Manager:
    def __init__(self, num_streams, data, labels, num_layers, num_unrollings, seg_len, verbose=False):
        self.num_streams    = num_streams
        self.data           = data
        self.labels         = labels
        self.num_layers     = num_layers
        self.num_unrollings = num_unrollings
        self.seg_len        = seg_len
        self.verbose        = verbose
        self.streams        = []
        self.segment_data()
        self.make_streams()

    def segment_data(self):
        seg_len    = len(self.data) / self.num_streams
        cursor     = 0
        out_data   = []
        out_labels = []
        for i in range(self.num_streams):
            out_data.append(  self.data[     seg_len*i:seg_len*(i+1)])
            out_labels.append(self.labels[:, seg_len*i:seg_len*(i+1)])
            cursor += seg_len
        self.data   = out_data
        self.labels = out_labels
        if self.verbose == True:
            print('created streams using ' + str(seg_len) + ' as seg_len')
            print('each dataset has length ' + str(len(self.data[0])))

    def make_streams(self):
        for i in range(self.num_streams):
            self.streams.append(Stream(self.data[i], self.labels[i], self.num_layers, self.num_unrollings, self.seg_len, verbose=self.verbose))


class Stream:
    def __init__(self, data, labels, num_layers, num_unrollings, seg_len, verbose=False):
        self.state  = np.zeros([num_layers, 2, num_unrollings, seg_len])
        self.data   = data
        self.labels = labels
        self.verbose = verbose
        if self.verbose == True:
            print('created a stream with ' + str(len(self.data)) + ' data and ' +str(len(labels)) + ' labels')


class Timer:
    def __init__(self, iters=0):
        self.prev_time   = time.time()
        self.prev_iters  = iters

    def evaluate(self, iters):
        new_time        = time.time()
        elapsed         = (new_time - self.prev_time) / float(iters - self.prev_iters)
        self.prev_time  = new_time
        self.prev_iters = iters
        return elapsed


class Control_Signal_Generator:
    '''takes in output length and format list and outputs 4 control signals.
       Two main modes: normal and separate notes.
       Available output formats:
            continuous : random envelope for the duration
            step       : step function with random integers and random lengths
            notes      : separate notes are generated with a single freq and hr for each note, continuous adrs for mi and am.
    '''
    def __init__(self, output_length, format_list, sampling_rate, seg_min = 11025 / 2, seg_max = 11025):
        self.output_length = output_length
        self.format_list   = format_list
        self.sampling_rate = sampling_rate
        self.seg_min       = seg_min
        self.seg_max       = seg_max
        self.min_midi      = 20
        self.max_midi      = 75

        if 'notes' in self.format_list:
            self.mode = 'notes'
        else:
            self.mode = 'normal'

    def normal_mode(self):
        output = []
        for control in self.format_list:
            if control == 'continuous':
                output.append(gen_random_envelope(self.output_length, self.seg_min, self.seg_max))
            elif control == 'step':
                output.append(random_step_function(self.min_midi, self.max_midi, self.output_length, same_length_steps=False, min_step_length=self.seg_min, max_step_length=self.seg_max, normalize=True))
            else:
                print('found invalid control format for normal_mode: ' + control)
        return np.array(output)

    def notes_mode(self):
        print('notes mode not yet implemented')
        return False

    def create_output(self):
        if self.mode == 'normal':
            self.current_data = self.normal_mode()
            return self.current_data
        else:
            self.current_data = self.notes_mode()
            return self.current_data

    def fm(self):
        return fm(self.current_data[0]*mtof(self.max_midi), self.current_data[1]*10.0, self.current_data[2], self.current_data[3], self.sampling_rate)



class Training_Manager:
    def __init__(self, lr              = 1e-4,
                        num_epochs     = 20,
                        seg_len        = 1024,
                        num_unrollings = 100,
                        sampling_rate  = 44100,
                        num_layers     = 1,
                        int_cursor_inc = 1,
                        ext_cursor_inc = 1,
                        weight_stddev  = 1e-6,
                        label_shape    = (1,1),
                        label_offset   = 1,
                        input_fn_list  = [None],
                        label_fn_list  = [None]
                        ):

        #hyper paramaters
        self.lr             = lr
        self.num_epochs     = num_epochs
        self.weight_stddev  = weight_stddev
        self.num_layers     = num_layers
        self.use_dropout    = False
        self.use_validation = False
        self.use_testing    = False
        self.use_residual   = False
        self.regularization = False
        self.reg_amount     = 0.5
        self.input_dropout  = 0.5
        self.output_dropout = 1.0
        self.noisify        = False
        self.sampling_rate  = sampling_rate
        
        #data shapes
        self.seg_len        = seg_len
        self.num_unrollings = num_unrollings
        self.int_cursor_inc = int_cursor_inc
        self.ext_cursor_inc = ext_cursor_inc
        self.label_shape    = label_shape
        self.label_offset   = label_offset
        self.output_shape   = label_shape
        self.use_transpose  = False


        #interface settings
        self.use_plotting   = False
        self.save_dreams    = True
        self.use_seq_dreams = True
        self.audio_freq     = 10000
        self.print_freq     = 1000
        self.plot_freq      = 5000
        self.dream_length   = self.sampling_rate * 30
        self.seq_dream_len  = self.sampling_rate * 5
        self.base_name      = 'outputs/basic_lstm_'
        self.comment        = ''
        self.verbose        = 1 #[0, 1, 2]

        #data file names
        self.input_fn_list  = input_fn_list
        self.label_fn_list  = label_fn_list


        #seeds
        self.seed_fn_list   = []
        self.seed_advance   = ['same', 'random', 'advance'] #'same' to always use same seed, 'advance' to move with train cursor, 'random' for random seeds
        self.seed_idx_offset= 0
        self.advance_cursor = 0
        self.seed_cursor    = 0


        #fft hyperparameters
        self.use_fft        = False
        self.hop_size       = self.seg_len
        self.window_size    = self.seg_len
        self.zero_padding   = True #double the size of the data with zeros, non-redudant form of data will be same length as window size


        #misc settings
        self.device         = '/gpu:' + str(np.random.randint(8))
        self.use_noise_state = False

    def save_json(self):
        output         = self.__dict__.copy()
        params_to_omit = ['device', 'num_iters', 'seed_list', 'input_data']

        #remove unwanted items from output dict
        for _ in params_to_omit:
            temp = output.pop(_)
 
        fn = self.base_name + self.comment + '.py'

        with open(fn, 'w') as f:
            f.write(str(output))
        print('saved ' + fn)

    def calc_noise(self, data, discount=0.0):
        return np.sum(np.square(np.diff(data))) - discount









    def read_in_data(self):
        #require self.label_fn_list to be a list [None] if no label is to be specified
        if len(self.input_fn_list) != len(self.label_fn_list):
            print('Found ' + str(len(self.input_fn_list)) + ' input file names but ' + str(len(self.label_fn_list) + ' label file names!'))
            quit()
        self.input_data = []
        sampling_rates  = []
        name_list       = []
        for fn in range(len(self.input_fn_list)):
            cur_input_fs, cur_input = wav.read(self.input_fn_list[fn])

            #if using fft, process data here
            if self.use_fft == True:
                cur_input    = break_into_chunks(cur_input, self.window_size, self.hop_size)#audio in chunks
                if self.zero_padding == True:
                    cur_input = to_real_better(cur_input)
                else:
                    cur_input = to_real(cur_input)

            sampling_rates.append(cur_input_fs)
            self.input_data.append(cur_input)
            name_list.append(self.input_fn_list[fn])
            if self.verbose >= 1:
                print('read in ' + self.input_fn_list[fn])

            #skip if the element in the label list is None
            if self.label_fn_list[fn] != None:
                cur_label_fs, cur_label = wav.read(self.label_fn_list[fn])
                #if using fft, process data here
                if self.use_fft == True:
                    cur_label    = break_into_chunks(cur_label, self.window_size, self.hop_size)#audio in chunks
                    if self.use_padding == True:
                        cur_label = to_real_better(cur_label)
                    else:
                        cur_label = to_real(cur_label)

                sampling_rates.append(cur_label_fs)
                self.labels.append(cur_input)
                name_list.append(self.label_fn_list[fn])
                if self.verbose >= 1:
                    print('read in ' + self.label_fn_list[fn])



        #=====read in or generate seed(s)=====#
        if len(self.seed_fn_list) > 0 and self.seed_fn_list != [None]:
            self.seed_list = []
            for fn in range(len(self.seed_fn_list)):
                if self.seed_fn_list[fn] == 'noise':
                    pass
                elif self.seed_fn_list[fn] == None:
                    pass
                else:
                    seed_fs, seed_data = wav.read(self.seed_fn_list[fn])

                    #if using fft, process data here
                    if self.use_fft == True:
                        seed_data    = break_into_chunks(seed_data, self.window_size, self.hop_size)#audio in chunks
                        if self.use_padding == True:
                            seed_data = to_real_better(seed_data)
                        else:
                            seed_data = to_real(seed_data)

                    sampling_rates.append(seed_fs)
                    name_list.append(self.seed_fn_list[fn])
                    self.seed_list.append(seed_data)
        else:
            self.seed_list = [self.input_data[0]]

        #=====check for sampling rate mismatches=====#
        fs_test = set(sampling_rates)
        if len(fs_test) != 1:
            print('Warning! Found sampling rate mismatch')
            for i in range(len(sampling_rates)):
                print(name_list[i] + ': ' + str(sampling_rates[i]))
        else:
            if self.sampling_rate != sampling_rates[0]:
                print('Warning! Training_Manager sampling rate does not match data sampling rate.')
                answer = raw_input('Change Training_Manager sampling rate from ' + str(self.sampling_rate) + ' to ' + str(sampling_rates[0]) + '? (True/False)')
                if answer == True or '1':
                    self.sampling_rate = sampling_rates[0]

        #=====calculate num_iters=====#
        self.train_cursor      = 0
        dataset_lengths        = np.array([len(i) for i in self.input_data])
        shortest               = np.argmin(dataset_lengths)
        if self.use_fft == False:
            len_matrix             = self.seg_len + (self.int_cursor_inc * (self.num_unrollings - 1))
            self.num_iters         = (len(self.input_data[shortest]) - len_matrix) / self.ext_cursor_inc
        else:
            self.num_iters         =  len(self.input_data[shortest]) - (self.num_unrollings + 1)
        if self.verbose >= 1:
            print('using ' + self.input_fn_list[shortest] + ' to calculate num_iters.')
            print('using num_iters = ' + str(self.num_iters))








    def next(self, source, advance=True):
        if self.use_fft == True:
            train = source[self.train_cursor:self.train_cursor+self.num_unrollings]
            label = source[self.train_cursor+self.label_offset:self.train_cursor+self.num_unrollings+self.label_offset]
            #return train, label
        else:
            train = []
            for i in xrange(self.num_unrollings):
                start = self.train_cursor+(i*self.int_cursor_inc)
                end   = start +self.seg_len
                train.append(source[start:end])
            if self.noisify != False:
                noise = np.random.randn(self.num_unrollings, self.seg_len) * self.noisify
                train = np.array(train) + noise

            label = []
            for j in xrange(self.label_shape[0]):
                start = self.train_cursor + self.label_offset + (j*self.int_cursor_inc)
                end   = start + self.label_shape[1]
                label.append(source[start:end])

        if advance == True:
            self.train_cursor += self.ext_cursor_inc
        return np.array(train), np.array(label)


    def prep_seed(self, force_mode=None):
        '''Cycle through all seeds and all methods of advancing the seed cursor.'''
        if   self.seed_advance[self.advance_cursor] == 'same':
            start = self.seed_idx_offset
            mode  = 'same'
        elif self.seed_advance[self.advance_cursor] == 'advance':
            start = self.train_cursor
            mode  = 'advance'
        elif force_mode != None:
            mode = force_mode
        elif self.seed_advance[self.advance_cursor] == 'random' and self.use_fft == False:
            if self.verbose == 2:
                print('inside random seed choice while loop')
            mode = 'random'
            done = False
            while done == False:
                start = np.random.randint(self.num_iters)
                if self.verbose == 2:
                    print('guess: ' + str(len(self.seed_list[self.seed_cursor][start:])) + ' must be less than target: ' + str(self.num_unrollings * self.seg_len))
                if len(self.seed_list[self.seed_cursor][start:]) >=  self.num_unrollings*self.seg_len:
                    done = True
        elif self.seed_advance[self.advance_cursor] == 'random' and self.use_fft == True:
            if self.verbose >= 1:
                print ('inside random seed choice while loop')
            mode = 'random'
            done = False
            while done == False:
                start = np.random.randint(self.num_iters)
                if len(self.seed_list[self.seed_cursor][start:]) >= self.num_unrollings+1:
                    done = True
                
        end   = start + (self.num_unrollings*self.seg_len)

        if self.verbose == 2:
            print('using ' + self.seed_advance[self.advance_cursor] + ' as mode to advance seed.')
            print('start: ' + str(start))
        if self.use_fft == True:
            output = self.seed_list[self.seed_cursor][start:start+self.num_unrollings]
        else:
            output = self.seed_list[self.seed_cursor][start:end].reshape((self.num_unrollings, self.seg_len))
        self.advance_cursor += 1
        self.advance_cursor %= len(self.seed_advance)
        self.seed_cursor    += 1
        self.seed_cursor    %= len(self.seed_list)

        return output, mode



#======================================================================#

if __name__ == '__main__':
	import numpy as np
	print('executing shared_functions.py')
