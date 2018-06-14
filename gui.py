import Tkinter      as tk
import ScrolledText as st


class Param:
    def __init__(self, name, value, dtype, comment, index):
        self.name    = name
        self.value   = value
        self.dtype   = dtype
        self.comment = comment

class Page:
    def __init__(self, preset, params, title, image_fn):
        self.preset = preset
        self.params = params
        self.title  = title
        self.image  = image_fn


class Preset:
    def __init__(self, name):
        self.name           = name
        self.lr             = Param('lr'             , 1e-4,                      float , 'learning rate'                                                       , 0)
        self.num_epochs     = Param('num_epochs'     , 20,                        int   , 'number of epochs'                                                    , 1)
        self.weight_stddev  = Param('weight_stddev'  , 1e-6,                      float , 'standard deviation to use for noise of layer weights initialization' , 7)
        self.num_layers     = Param('num_layers'     , 1,                         int   , 'number of lstm layers'                                               , 4)
        self.use_dropout    = Param('use_dropout'    , 0.0,                       float , 'amount to use for dropout. 0 for no dropout.'                        , 14)
        self.use_residual   = Param('use_residual'   , 'False',                   bool  , 'whether or not to use residual wrapper on lstm'                      , 15)
        self.regularization = Param('regularization' , 'False',                   bool  , 'whether or not to use lstm internal regularization'                  , 16)
        self.reg_amount     = Param('reg_amount'     , 0.5,                       float , 'amount of regularization to apply'                                   , 16)
        self.input_dropout  = Param('input_dropout'  , 0.5,                       float , 'percent chance to keep the connection'                               , 16)
        self.output_dropout = Param('output_dropout' , 1.0,                       float , 'percent chance to keep the connection'                               , 16)
        self.seg_len        = Param('seg_len'        , 1024,                      int   , 'length of each input vector'                                         , 2)
        self.num_unrollings = Param('num_unrollings' , 100,                       int   , 'number of input vectors per time step'                               , 3)
        self.int_cursor_inc = Param('int_cursor_inc' , self.seg_len.value,        int   , 'difference between first index of adjacent input vectors'            , 5)
        self.ext_cursor_inc = Param('ext_cursor_inc' , 1,                         int   , 'difference between first index of subsequent input matrices'         , 6)
        self.label_shape_0  = Param('label_shape_0'  , self.num_unrollings.value, int   , 'number of elements in output, label'                                 , 9)
        self.label_shape_1  = Param('label_shape_1'  , self.seg_len.value,        int   , 'length of each element in output, label'                             , 10)
        self.label_shape    = Param('label_shape'    , (1, 1),                    tuple , 'shape of the label'                                                  , 10)
        self.label_offset   = Param('label_offset'   , self.seg_len.value,        int   , 'difference between first index of input and first index of label'    , 11)
        self.use_transpose  = Param('use_transpose'  , False,                     bool  , 'whether or not to apply addition fully connect layer'                , 11)
        self.sampling_rate  = Param('sampling_rate'  , 11025,                     int   , 'sampling rate of input wav files and output wav files'               , 8)
        self.use_plotting   = Param('use_plotting'   , 'False',                   bool  , 'whether or not to use plotting during training'                      , 12)
        self.save_dreams    = Param('save_dreams'    , 'True',                    bool  , 'whether or not to make predictions during training'                  , 13)
        self.audio_freq     = Param('audio_freq'     , 10000,                     int   , 'number of iterations to wait before making prediction'               , 17)
        self.print_freq     = Param('print_freq'     , 1000,                      int   , 'how many iterations to wait before printing the loss again'          , 22)
        self.dream_length   = Param('dream_length'   , 400,                       int   , 'how many sampling iterations (sometimes samples, sometimes vectors)' , 22)
        self.seq_dream_len  = Param('seq_dream_len'  , 400,                       int   , 'like dream_length but for sequential dreams'                         , 22)
        self.comment        = Param('comment'        , '_',                       str   , 'comment to add to output file names'                                 , 18)
        self.input_fn_list  = Param('input_fn_list'  , ['model_inputs/data.wav'], list  , 'list of input file names'                                            , 19)
        self.label_fn_list  = Param('label_fn_list'  , ['model_inputs/data.wav'], list  , 'list of label file names'                                            , 20)
        self.seed_fn_list   = Param('seed_fn_list'   , [None],                    list  , 'list of seed file names'                                             , 21)
        self.use_fft        = Param('use_fft'        , False,                     bool  , 'whether or not to apply fft to data'                                 , 21)
        self.hop_size       = Param('hop_size'       , 256 ,                      int   , 'hop size to use when taking fft'                                     , 21)
        self.window_size    = Param('window_size'    , 1024,                      int   , 'window size to use when taking fft'                                  , 21)
        self.zero_padding   = Param('zero_padding'   , True,                      bool  , 'whether or not to pad fft windows with zeros'                        , 21)

        self.params = [self.lr,
                       self.num_epochs,
                       self.weight_stddev,
                       self.num_layers,
                       self.use_dropout,
                       self.use_residual,
                       self.regularization,
                       self.reg_amount,
                       self.input_dropout,
                       self.output_dropout,
                       self.seg_len,
                       self.num_unrollings,
                       self.int_cursor_inc,
                       self.ext_cursor_inc,
                       self.label_shape_0,
                       self.label_shape_1,
                       self.label_shape,
                       self.label_offset,
                       self.use_transpose,
                       self.sampling_rate,
                       self.use_plotting,
                       self.save_dreams,
                       self.audio_freq,
                       self.print_freq,
                       self.dream_length,
                       self.seq_dream_len,
                       self.comment,
                       self.input_fn_list,
                       self.label_fn_list,
                       self.seed_fn_list,
                       self.use_fft,
                       self.hop_size,
                       self.window_size,
                       self.zero_padding]

class Vector_Preset(Preset):
    def __init__(self):
        Preset.__init__(self, 'vector')
        self.label_shape.value   = (self.num_unrollings.value, self.seg_len.value)
        self.use_fft.value       = False
        self.use_transpose.value = False

class Magnitude_Preset(Preset):
    def __init__(self):
        Preset.__init__(self, 'magnitude')
        self.use_transpose.value = False
        self.label_shape.value   = (self.num_unrollings.value, self.seg_len.value)



class Transpose_Preset(Preset):
    def __init__(self):
        Preset.__init__(self, 'transpose')
        self.label_shape.value   = (1, 1)
        self.use_fft.value       = False
        self.use_transpose.value = True

class Column_Preset(Preset):
    def __init__(self):
        Preset.__init__(self, 'column')
        self.label_shape.value   = (self.num_unrollings.value, 1)
        self.use_fft.value       = False
        self.use_transpose.value = False
        self.use_dropout.value   = True




class Main_Program:
    def __init__(self, root):
        #________style________#
        self.title_font      = ('Futura Medium BT', '20')#('Rockwell')#("URW\ Gothic\ L,\ Demi", "18", "bold")
        self.code_font       = ('Courier\ 10\ Pitch', '12')
        self.number_font     = ('FreeMono')
        self.light_font      = ('Futura\ Light\ Condensed\ BT', '12')#("URW\ Gothic\ L,\ Book", "12")

        self.window_width    = "1200"
        self.window_height   = "600"
        self.label_bkgd      = 'white'



        # self.params = [self.lr,
        #                self.num_epochs,
        #                self.seg_len,
        #                self.num_unrollings,
        #                self.num_layers,
        #                self.int_cursor_inc,
        #                self.ext_cursor_inc,
        #                self.weight_stddev,
        #                self.sampling_rate,
        #                self.label_shape_0,
        #                self.label_shape_1,
        #                self.label_shape,
        #                self.label_offset,
        #                self.use_plotting,
        #                self.save_dreams,
        #                self.use_dropout,
        #                self.use_residual,
        #                self.regularization,
        #                self.audio_freq,
        #                self.comment,
        #                self.input_fn_list,
        #                self.label_fn_list,
        #                self.seed_fn_list,
        #                self.print_freq
        #                ]

        root.title("Job Manager")
        root.geometry(self.window_width + 'x' + self.window_height)
        root.configure(background='white')

        self.form_title_page()

    def form_title_page(self):
        self.clear()
        self.page_title = tk.Label(root, text='Choose A Preset', font=self.title_font , background=self.label_bkgd)#, height=self.title_height)

        self.page_title.grid(column=0, row=1, columnspan=4)

        self.vector_button    = tk.Button(root, text='Vector',             command= lambda: self.form_vector_page('Vector'),             font=self.title_font, background='white')
        self.magnitude_button = tk.Button(root, text='Magnitude Spectrum', command= lambda: self.form_vector_page('Magnitude Spectrum'), font=self.title_font, background='white')
        self.transpose_button = tk.Button(root, text='Transpose',          command= lambda: self.form_vector_page('Transpose'),          font=self.title_font, background='white')
        self.column_button    = tk.Button(root, text='Column',             command= lambda: self.form_vector_page('Column'),             font=self.title_font, background='white')

        self.vector_button.grid(   column=0, row=2)
        self.magnitude_button.grid(column=1, row=2)
        self.transpose_button.grid(column=2, row=2)
        self.column_button.grid(   column=3, row=2)

    def form_vector_page(self, title_text):
        self.clear()
        self.page_title      = tk.Label(root, text=title_text, font=self.title_font , background=self.label_bkgd)#, height=self.title_height)

        if title_text == 'Vector':
            self.preset_instance = Vector_Preset()
        elif title_text == 'Magnitude Spectrum':
            self.preset_instance = Magnitude_Preset()
        elif title_text == 'Transpose':
            self.preset_instance = Transpose_Preset()
        elif title_text == 'Column':
            self.preset_instance = Column_Preset()


        self.text_box = tk.Text(root, height=40)
        self.text_box.grid(row=1, column = 0, rowspan = 34, columnspan=3)
        self.comments = tk.Text(root, height=40)
        self.comments.grid(row=1, column = 4, rowspan = 34, columnspan=3)

        # self.process_output()

        string_version = ''
        comment_string = ''



        for p in range(len(self.preset_instance.params)):
            name           = self.preset_instance.params[p].name
            comment        = self.preset_instance.params[p].comment
            padded         = ' '*(14-len(name)) + name
            string_version = string_version + padded + ' : ' + str(self.preset_instance.params[p].value) + ',' + '\n'
            comment_string = comment_string + comment + '\n'
        self.text_box.insert(tk.INSERT, string_version)
        self.comments.insert(tk.INSERT, comment_string)
        self.comments.configure(state='disabled')


        self.back_button    = tk.Button(root, text='back',  command= lambda:  self.form_title_page(), font=self.title_font, background='white')
        self.quit_button    = tk.Button(root, text='quit',  command= lambda:  self.quit(),            font=self.title_font, background='white')
        self.save_button    = tk.Button(root, text='save',  command= lambda:  self.save(),            font=self.title_font, background='white')
        self.back_button.grid(column=0, row=31)
        self.save_button.grid(column=1, row=31)
        self.quit_button.grid(column=2, row=31)

    def process_output(self):

        self.get_entries()
        self.output = {}
        for param in self.preset_instance.params:
            if type(param.value) == param.dtype:
                self.output[param.name] = param.value
            else:
                try:
                    val = eval(param.value)
                    if type(val) == param.dtype:
                        self.output[param.name] = val
                    else:
                        print('could not convert to correct data type')
                        print(param.name, type(val), param.dtype)
                except:
                    print('found incorrect data type')
                    print(param.name + str(param.value) + ' has type ' + str(type(param.value)) + ' but should have type ' + str(param.dtype))

        #handle label shape differently
        self.preset_instance.label_shape = (int(self.preset_instance.label_shape_0.value), int(self.preset_instance.label_shape_1.value))
        self.output['label_shape']       = self.preset_instance.label_shape

    def next(self):
        print('next')
        self.get_entries()

    def clear(self):
        print('clear')
        widget_list = root.grid_slaves()
        for l in widget_list:
            l.destroy()

    def quit(self):
        print('quit')
        quit()

    def get_entries(self):
        output  = {}
        entries = self.text_box.get("1.0",'end-1c').split('\n')#read in the data from the gui
        entries = [str(i).split(':') for i in entries]         #split it into key value pairs
        entries = entries[:-1]                                 #get rid of extra entry
        for e in range(len(entries)):
            key          = ' '.join(entries[e][0].split())     #remove extra leading spaces from key
            target_type  = self.preset_instance.params[e].dtype#figure out what the datatype is supposed to be
            #if it's not supposed to be a string
            if type(entries[e][1]) != target_type:
                if target_type == tuple:
                    val = eval(entries[e][1][:-1])
                elif target_type == list:
                    val = eval(entries[e][1][:-1])
                elif target_type == str:
                    val = entries[e][1]
                else:
                    val = target_type(entries[e][1][:-1])
            output[key]  = val
        self.output = output


    def save(self):
        self.get_entries()
        fn = 'temp.py'
        with open(fn, 'w') as f:
            f.write('data = ' + str(self.output))
        print('saved ' + fn)
        self.pop_up = tk.Toplevel(root)
        tk.Label(self.pop_up, text='Saved hyperparameters into the file \'temp.py\''    ).pack()
        self.ok_button = tk.Button(self.pop_up, text='ok',   command=self.pop_up.destroy).pack()
        self.save_quit = tk.Button(self.pop_up, text='quit', command=self.quit          ).pack()




#================================================================#
if __name__ == '__main__':
    root  = tk.Tk()
    main  = Main_Program(root)

    v = Vector_Preset()
    print(v.label_shape.value)

    root.mainloop()
