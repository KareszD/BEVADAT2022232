# Create a function that returns with a subset of a list.
# The subset's starting and ending indexes should be set as input parameters (the list aswell).
# return type: list
# function name must be: subset
# input parameters: input_list,start_index,end_index

def subset(input_list:list,start_index,end_index):
    output_list = input_list[start_index:end_index]
    return output_list

# Create a function that returns every nth element of a list.
# return type: list
# function name must be: every_nth
# input parameters: input_list,step_size

def every_nth(input_list,step_size):
    output_list = input_list[0:len(input_list):step_size]
    return output_list

#test
#t = (1,2,3,4,5,6,7,8,9,10)
#print(every_nth(x,3))

# Create a function that can decide whether a list contains unique values or not
# return type: bool
# function name must be: unique
# input parameters: input_list

def unique(input_list):
    return len(input_list) == len(set(input_list))

# Create a function that can flatten a nested list ([[..],[..],..])
# return type: list
# fucntion name must be: flatten
# input parameters: input_list

def flatten(input_list):
    output_list = list()
    for elemento in input_list:
        for elementi in elemento:
            output_list.append(elementi)
    return output_list

#test
#t = ['a', ['bb', 'ee', 'ff'], 'g', 'h']
#print(flatten(t))


# Create a function that concatenates n lists
# return type: list
# function name must be: merge_lists
# input parameters: *args

def merge_lists(*args):
    output_list = list()
    for element in args:
        output_list += element
    return output_list

# Create a function that can reverse a list of tuples
# example [(1,2),...] => [(2,1),...]
# return type: list
# fucntion name must be: reverse_tuples
# input parameters: input_list

def reverse_tuples(input_list):
    output_list = list()
    for element in input_list:
        output_list.append(reversed(element))
    return output_list

# Create a function that removes duplicates from a list
# return type: list
# function name must be: remove_tuplicates
# input parameters: input_list

def remove_tuplicates(input_list):
    return list(set(input_list))

# Create a function that transposes a nested list (matrix)
# return type: list
# function name must be: transpose
# input parameters: input_list

def transpose(input_list):
    output_list = list()
    for row in range(len(input_list[0])):
        column = list()
        for element in input_list:
            column.append(element[row])
        output_list.append(column)
    return output_list

#test
#t = [[1, 4, 5, 12],
    #[-5, 8, 9, 0],
    #[-6, 7, 11, 19]]
#print(transpose(t))

# Create a function that can split a nested list into chunks
# chunk size is given by parameter
# return type: list
# function name must be: split_into_chunks
# input parameters: input_list,chunk_size

def split_into_chunks(input_list,chunk_size):
    temp_list = list()
    for element in input_list:
        temp_list += element

    output_list = list()
    for element in range(0,len(temp_list),chunk_size):
        output_list.append(temp_list[element:element+chunk_size])
    return output_list

# Create a function that can merge n dictionaries
# return type: dictionary
# function name must be: merge_dicts
# input parameters: *dict

def merge_dicts(*dict):
    output_dict = {}
    for element in dict:
        output_dict.update(element)
    return output_dict

# Create a function that receives a list of integers and sort them by parity
# and returns with a dictionary like this: {"even":[...],"odd":[...]}
# return type: dict
# function name must be: by_parity
# input parameters: input_list

def by_parity(input_list):
    output_dict = {"even":[],"odd":[]}
    for element in input_list:
        if element % 2 == 0:
            output_dict["even"].append(element)
        else:
            output_dict["odd"].append(element)
    return output_dict

# Create a function that receives a dictionary like this: {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....}
# and return a dictionary like this : {"some_key":mean_of_values,"another_key":mean_of_values,....}
# in short calculates the mean of the values key wise
# return type: dict
# function name must be: mean_key_value
# input parameters: input_dict

def mean_key_value(input_dict):
    output_dict = {}
    for element in input_dict.keys:
        output_dict[element] = sum(input_dict[element])/len(input_dict[element])
    return output_dict
# If all the functions are created convert this notebook into a .py file and push to your repo