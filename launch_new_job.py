output_dict  = {}

param_dict   = {0  : ['lr'             , 1e-4,                            float , 'learning rate'                                                                 ],
                1  : ['num_epochs'     , 20,                              int   , 'number of epochs'                                                              ],
                2  : ['seg_len'        , 1024,                            int   , 'length of each input vector'                                                   ],
                3  : ['num_unrollings' , 100,                             int   , 'number of input vectors per time step'                                         ],
                4  : ['num_layers'     , 1,                               int   , 'number of lstm layers'                                                         ],
                5  : ['int_cursor_inc' , 'output_dict["seg_len"]',        int   , 'difference between first index of adjacent input vectors'                      ],
                6  : ['ext_cursor_inc' , 1,                               int   , 'difference between first index of subsequent input matrices'                   ],
                7  : ['weight_stddev'  , 1e-6,                            float , 'standard deviation to use for noise of layer weights initialization'           ],
                8  : ['sampling_rate'  , 11025,                           int   , 'sampling rate of input wav files and output wav files'                         ],
                9  : ['label_shape_0'  , 'output_dict["num_unrollings"]', int   , 'number of elements in output, label'                                           ],
                10 : ['label_shape_1'  , 'output_dict["seg_len"]',        int   , 'length of each element in output, label'                                       ],
                11 : ['label_offset'   , 'output_dict["seg_len"]',        int   , 'difference between first index of input matrix and first index of label matrix'],
                12 : ['use_plotting'   , False,                           bool  , 'whether or not to use plotting during training'                                ],
                13 : ['save_dreams'    , True,                            bool  , 'whether or not to make predictions during training'                            ],
                14 : ['use_dropout'    , False,                           float , 'amount to use for dropout'                                                     ],
                15 : ['use_residual'   , False,                           bool  , 'whether or not to use residual wrapper on lstm'                                ],
                16 : ['regularization' , False,                           bool  , 'whether or not to use lstm internal regularization'                            ],
                17 : ['audio_freq'     , 10000,                           int   , 'number of iterations to wait before making prediction'                         ],
                18 : ['comment'        , '',                              str   , 'comment to add to output file names'                                           ],
                19 : ['input_fn_list'  , ['model_inputs/data.wav'],       list  , 'list of input file names'                                                      ],
                20 : ['seed_fn_list'   , [None],                          list  , 'list of seed file names'                                                       ]
                }





def get_input(param, default):
    '''get input from user, if nothing, use default, santize input, repeat if necessary'''
    done        = False
    while done == False:
        try:
            #get input from user
            answer      = input(str(param) + ' :')
        except:
            #if the input is '' use the default value, if default is to be evaluated, eval()
            if 'output_dict' in str(default):
                default = eval(default)
            print(str(param) + ' = ' + str(default))
            #allows for use of params as variable names in later calls to get_input()
            exec("param + ' = ' + str(default)")
            return default
        if type(answer) == type(default):
            #if the input type is correct done
            done = True
        else:
            #if the input is the wrong type, try again
            print('invalid type expecting ' + str(type(default)) + ' but got ' +  str(type(answer)))
    #allows for use of params as variable names in later calls to get_input()
    exec(param + ' = ' + str(answer))
    return answer


def file_exists(fn):
    try:
        f = open(fn)
        return True
    except:
        return False

def replace_bad_fns(input_list):
    '''find bad filenames in input lists, try to replace them'''
    missing_fns = []
    for fn in input_list:
        if file_exists(fn) == False:
            missing_fns.append(fn)
    for fn in missing_fns:
        done = False
        while done == False:
            print('could not find file: ' + str(fn))
            answer = raw_input('replace with>> ')
            if '"' in answer or "'" in answer:
                print('\n'*2 + 'Do not include single or double quotes in file name!' + '\n'*2)
            if file_exists(answer) == True:
                print('found ' + answer + ' replacing ')
                done = True
                input_list.remove(fn)
                input_list.append(answer)
            else:
                print('could not find file: ' + answer)


def finalize(output_dict):
    output_dict['label_shape'] = (output_dict['label_shape_0'], output_dict['label_shape_1'])
    replace_bad_fns(output_dict['input_fn_list'])
    replace_bad_fns(output_dict['seed_fn_list'])



if __name__ == '__main__':
    for p in range(len(param_dict)):
        output_dict[param_dict[p][0]] = get_input(param_dict[p][0], param_dict[p][1])


    finalize(output_dict)

    print output_dict
