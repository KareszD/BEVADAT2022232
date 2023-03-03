# Create a function that decides if a list contains any odd numbers.
# return type: bool
# function name must be: contains_odd
# input parameters: input_list

def contains_odd(input_list):
    odd = False
    for element in input_list:
        if element % 2 == 1:
            odd = True
    return odd

# Create a function that accepts a list of integers, and returns a list of bool.
# The return list should be a "mask" and indicate whether the list element is odd or not.
# (return should look like this: [True,False,False,.....])
# return type: list
# function name must be: is_odd
# input parameters: input_list

def is_odd(input_list):
    output_list = list()
    for element in input_list:
        if element % 2 ==1:
            output_list.append(True)
        else:
            output_list.append(False)
    return output_list

# Create a function that accepts 2 lists of integers and returns their element wise sum.
# (return should be a list)
# return type: list
# function name must be: element_wise_sum
# input parameters: input_list_1, input_list_2

def element_wise_sum(input_list_1, input_list_2):
    output_list = list()
    i = -1
    if len(input_list_1) != 0 or len(input_list_2) != 0:
        for i in range(min(len(input_list_1),len(input_list_2))):
            output_list.append(input_list_1[i] + input_list_2[i])
    if i < len(input_list_1):
        for j in range(i+1,len(input_list_1)):
            output_list.append(input_list_1[j])
    if i < len(input_list_2):
        for j in range(i+1, len(input_list_2)):
            output_list.append(input_list_2[j])
    return output_list

#test
#list1 = [22,11,33,66]
#list2 = [11, 25]
#print(element_wise_sum(list1,list2))

# Create a function that accepts a dictionary and returns its items as a list of tuples
# (return should look like this: [(key,value),(key,value),....])
# return type: list
# function name must be: dict_to_list
# input parameters: input_dict

def dict_to_list(input_dict: dict):
    output_list = list()
    for key,value in input_dict.items():
        output_list.append((key,value))
    return output_list

# If all the functions are created convert this notebook into a .py file and push to your repo