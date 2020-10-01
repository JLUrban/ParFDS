import os, itertools, shutil, re, string, sys
import numpy as np

def dict_product(dicts):
	"""
	dict_product(dicts)

	from a dict of parameters creates a generator which outputs a list of dicts 
	effectively a cartesian product over the parameter space.

	eg: from:  	{'a': [0,1], 'b': [2,3], 'c': [4]}

	outputs:   [{'a': 0, 'b': 2, 'c': 4},
				{'a': 0, 'b': 3, 'c': 4},
				{'a': 1, 'b': 2, 'c': 4},
				{'a': 1, 'b': 3, 'c': 4}]
	"""
	# from http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
	return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def dict_builder(params, test_name = ''): 
	"""
	dict_builder(params)

	uses the dict_product function and adds a 
	title key value pair for use in the input files
	
	eg: from:  	{'a': [0,1], 'b': [2,3], 'c': [4]}

	outputs:    [{'TITLE': 'STEP_BOX0-4-2', 'a': '0', 'b': '2', 'c': '4'},
				 {'TITLE': 'STEP_BOX0-4-3', 'a': '0', 'b': '3', 'c': '4'},
				 {'TITLE': 'STEP_BOX1-4-2', 'a': '1', 'b': '2', 'c': '4'},
				 {'TITLE': 'STEP_BOX1-4-3', 'a': '1', 'b': '3', 'c': '4'}]
	"""

	for value_set in dict_product(params):
		title = "-".join(map(str, value_set.values())).replace('.', '_')
		vals = [dict([a, str(x)] for a, x in value_set.items())]
		vals = vals[0]
		vals['TITLE'] = test_name + title
		yield vals

def input_directory_builder(folder_name, base_path):
	"""
	input_directory_builder(folder_name, base_path)

	taking a base_path and a particular folder (folder_name) and creating both 
	folder_name inside of base_path if base_path does not exist and if base_path
	does exist, just creates folder_name inside of it.
	"""
	calling_dir = os.getcwd()
	
	if not os.path.exists(os.path.join(calling_dir, base_path)):
		os.mkdir(base_path)

	try:
		os.chdir(os.path.join(calling_dir, base_path))
		os.mkdir(folder_name)
	except:
		raise
	finally:
		os.chdir(calling_dir)

def build_input_files(filename, base_path = 'input_files', out = sys.stdout):
    """
    build_input_files(filename, base_path = 'input_files')
    
    takes a 'well-formated' input fileand outputs a 
    directory structure with the properly formated input files
    created in them. 
    """
    calling_dir = os.getcwd()
    
    # I'm doing this because I need it later
    file_path, file_name = os.path.split(filename)
    
    with open(filename, 'r') as f:
        txt = f.read()
    
    param_dict, txt, IOoutput, axes = FDSa_parser(txt, file_name, out)
    # param_dict, sweep_param_dict, prms_in_axis = caculate_params(param_dict, axes)
    print("Here Here")
    formatted_trials, logfile, IOoutput = eval_parsed_FDS(param_dict, axes, out)
        
    for i, value_set in enumerate(formatted_trials):
        tmp_txt = txt
        # make a directory
        case_name = 'case_'+int2base(i, 26)
        # FDS uses uppercase reseved keywords, and so will we
        value_set['TITLE'] = case_name
        input_directory_builder(case_name, base_path)
        # populate the input file
        tmp_txt = tmp_txt.format(**value_set)
        # create the file name
        fname = os.path.join(calling_dir, base_path, 
                    case_name, case_name + '.fds')
        # write the input file to the directory
        with open(fname, 'w') as f:
            f.write(str(tmp_txt))
        
    log_path_name = os.path.join(calling_dir, base_path, file_name[:-4] + '.log')
    
    # write the augmented fds log file

    with open(log_path_name, 'a') as f:
        f.write(logfile)
        
    return IOoutput
    
def input_file_paths(base_path):
	"""
	input_file_paths(base_path) 

	returns the paths of the input files present by recusivly walking over them
	so they can be iterated over in the main body of the program via multiprocessing
	"""
	paths = []
	for dirpath, dirnames, filenames in os.walk(base_path):
		for onefile in filenames:
			# the following if statement is due to OS X .DsStore bullshit...
			if not (onefile.startswith('.DS') or onefile.endswith('.log')):
				#paths.append(dirpath+"/"+onefile)      
				paths.append(os.path.join(os.getcwd(), dirpath, onefile))
	return paths
	
def int2base(x, base=26):
    """
    int2base(x, base) 

    takes an integer and returns the base 26 representation (defualt) in letters
    like one would see in excel column labeling (0 -> a, 63 -> cl)
    
    based on https://stackoverflow.com/questions/2267362
    """
    digs = string.ascii_lowercase
    assert type(x) is int, "x is not an integer: %r" % x
    assert type(base) is int, "base is not an integer: %r" % base
    
    if x < 0: sign = -1
    elif x==0: return 'a'
    else: sign = 1
    x *= sign
    digits = []
    y = x
    while y:
        print(x % base)
        digits.append(digs[x % base])
        y = x // base
        x = (x // base) - 1
    if sign < 0:
        digits.append('-')
        digits.reverse()
    # 
    return ''.join(digits)[::-1]

def FDSa_parser(txt, filename, IOoutput=sys.stdout):
    """
    FDSa_parser(txt, filename, IOoutput) 

    takes in an augmented FDS file and determines how many 
    parametric will be created from that it also parses the augmented syntax to 
    build the dictionary used in generating the specific case FDS files
    """
    ## I'm doing this because I need it later
    #file_path, file_name = os.path.split(filename)
    # open the augmented fds input file
    #with open(os.path.join(file_path, file_name), 'r') as f:
    #    read_data = f.read()
        
    regex_find = re.findall('\{*[0-9a-zA-Z_:,.\s]*\}', txt)
    
    params = []
    params_raw = []
    axes = []

    for param in regex_find:
        params_raw.append(param.strip('{}'))
        params.append(param.strip('{}').split('SWEEP'))

        if "TITLE" not in param:
            if 'axis' in param:
                params[-1][-1] = params[-1][-1].split('axis')[0] 
                axes.append(param.strip('{}').split('axis')[1])
            else:
                axes.append(None)

        # print(params[-1])
    print("axes")
    # print(axes)
 
      


   
    params = [item.strip() for sublist in params for item in sublist]
    # print(params) 
    



    # if length of params is non-even that means I can assume a title parameter
    # double check with the occurance of FDSa 'reserved' keywords 'title' or 'name'
    if (len(params) % 2 != 0) and (params[0].lower() == ('title')):
        # based on the following idiom 
        # https://stackoverflow.com/questions/3303213
        param_dict = dict(zip(params[1::2], params[2::2]))
        param_list = [params[0]]
        param_list.extend(params[1::2])
        param_name_dict = dict(zip(param_list, params_raw))
    else:
        param_dict = dict(zip(params[::2], params[1::2]))
        param_list = params[::2]
        param_name_dict = dict(zip(param_list, params_raw))
    
    # dealing with the `:` and `.` issue in the FDS file due to 
    # key value restrictions in python 
    for key, value in param_name_dict.items():
        #txt = string.replace(txt, value, key)
        txt = txt.replace(value, key)

    IOoutput.write('-'*10 + 'ParFDS input file interpreter' + '-'*10 + '\n')
    IOoutput.write('the following are the keys and values'+ '\n')
    IOoutput.write('seen in ' + filename + '\n')

    #print(param_dict)
    #assert False, "DEBUG"   

 
    return param_dict, txt, IOoutput, axes


def assign_dimensions(param_dict, axes):
    
    ## Assign Dimensions First
    axes = np.array(axes)
    reserved_dims = np.unique(axes[axes != None]).astype(int)
    axis_list = []
    prms_in_axis = {}
    cnt = 0
    for i, (ax, prm) in enumerate(zip(axes,param_dict.keys())):
#         print(i, (ax, prm))
        if ax is None:
            while (cnt in axis_list) or (cnt in reserved_dims):
                cnt += 1
            axis_list.append(cnt)
            prms_in_axis[axis_list[-1]] = [prm,]
            
        elif int(ax) in axis_list:
#             print("here")
            prms_in_axis[int(ax)].append(prm)

        else:
            axis_list.append(int(ax))
            prms_in_axis[axis_list[-1]] = [prm,]
#     print(prms_in_axis)
    return prms_in_axis
            
def caculate_params(param_dict, axes):
    ## Calculate Values
    print('here')
    prms_in_axis = assign_dimensions(param_dict, axes)
    permutations = 1
    for key, value in param_dict.items():
        value_str = 'np.linspace(' + value.replace("'", "") +')'
        param_dict[key] = eval(value_str, {"__builtins__":None}, 
                              {"np": np,"np.linspace":np.linspace,"np.logspace":np.logspace})
        value_split = value.split(',')
        
        assert float(value_split[2]) >= 1, "the number of steps is not an integer: %r" % float(value_split[2].strip())
        
        
    # Check to make sure parameters on the same axis have the same number of steps
    sweep_param_dict = {}
    for ax in prms_in_axis.keys():
        if len(prms_in_axis[ax]) > 1:
            sweep_param_dict["CONSTRAINED_"+str(ax)] = np.arange(len(param_dict[prms_in_axis[ax][0]]))
            for i , prm in enumerate(prms_in_axis[ax]):
                if i == 0:
                    # Store the first number of steps
                    test = len(param_dict[prm])
                else:
                    # Check other number of steps against the first
                    assert test == len(param_dict[prm]), "parameters on the same axis must have number of steps"
                    
                    
        else:
            sweep_param_dict[prms_in_axis[ax][0]] = param_dict[prms_in_axis[ax][0]]
            
            
    # print('HEREHEHRHEHERH')
    # print(sweep_param_dict)
    return param_dict, sweep_param_dict, prms_in_axis


def eval_parsed_FDS(param_dict, axes, IOoutput = sys.stdout):       
    """
    eval_parsed_FDS(param_dict, IOoutput = sys.stdout) 

    takes the dictionary that is returned by FDSa_parser and actually evaluates 
    it to create python readable arrays that can be broken up for the parametric studies.
    """
    # permutations = 1
    # for key, value in param_dict.items():
    #     value_str = 'np.linspace(' + value.replace("'", "") +')'
    #     param_dict[key] = eval(value_str, {"__builtins__":None}, 
    #                           {"np": np,"np.linspace":np.linspace,"np.logspace":np.logspace})
    #     value_split = value.split(',')
        
    #     assert float(value_split[2]) >= 1, "the number of steps is not an integer: %r" % float(value_split[2].strip())
        
    #     permutations *= int(value_split[2])
        
    #     IOoutput.write(key + ' varied between ' + str(value_split[0]) +\
    #         ' and ' + str(value_split[1]) + ' in ' + str(value_split[2]) + ' step(s)' + '\n')
    
    # IOoutput.write('-'*10 + ' '*10 + '-'*10 + ' '*10 + '-'*10 + '\n') 
    # IOoutput.write('for a total of ' + str(permutations) + ' trials' + '\n')
    # print("entering... eval_parsed_FDS")
    param_dict, sweep_param_dict, prms_in_axis = caculate_params(param_dict, axes)
    # print(param_dict)
    # assert False,'ASSERT'
    trials = dict_product(sweep_param_dict)

    formatted_trials = []
    permutations = 1
    for vs in sweep_param_dict.values():
        permutations *= len(vs)    
    logfile = 'There are a total of ' + str(permutations) + ' trials \n'
    newline = '\n' # for the newlines

    print("permutations: "+str(permutations))

    base = 26
    for i, trial_ in enumerate(trials):
        ## JLU added in
        print("i: "+str(i))
        o_keys = trial_.keys()
        for key in list(o_keys):
            if key.count("CONSTRAINED_")==1:
                ii = trial_.pop(key)
                for prm in prms_in_axis[int(key.split("_")[1])]:
                    trial_[prm] = param_dict[prm][ii]
        ## \end
        case_temp = 'case ' + int2base(i, base) + ': '
        logfile += case_temp
        IOoutput.write(case_temp,)
        for key, val in trial_.items():
            kv_temp = key + ' ' + str(round(val, 2)) + ' '
            logfile += kv_temp + ' '
            IOoutput.write(kv_temp,)
        IOoutput.write(newline)
        logfile += newline
        formatted_trials.append({key : value for key, value in trial_.items() })

    
    """
    >>> important_dict = {'x':1, 'y':2, 'z':3}
    >>> name_replacer = {'x':'a', 'y':'b', 'z':'c'}
    >>> {name_replacer[key] : value for key, value in important_dict.items() }
    {'a': 1, 'b': 2, 'c': 3}
    """    
    # IOoutput.getvalue().strip()
    return formatted_trials, logfile, IOoutput
