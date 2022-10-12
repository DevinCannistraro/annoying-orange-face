import random

# return raw 2d array from csv
def read_csv(file_path, delimiter,chars_to_remove):
    with open(file_path) as fp:
        Lines = fp.readlines()
        data = []
        for line in Lines:
            line = line.strip()
            for char in chars_to_remove:
                line = line.replace(char,'')
            data.append(line.split(delimiter))
    return data

def get_first_row_label_data(file_path,delimiter):
    data = read_csv(file_path,delimiter,['\"'])
    header = data[0]
    formatted_data = []
    for entry in data[1:]:
        formatted_entry = {}
        for index,value in enumerate(entry):
            formatted_entry[header[index]] = value
        formatted_data.append(formatted_entry)
    return formatted_data

def get_n_random_entries(n,formatted_data):
    choices = []
    while n > 0:
        choice = random.choice(formatted_data)
        if choice not in choices: # only add if not already in list
            choices.append(choice)
            n -= 1
    return choices


def bubbleSort(arr,target_column):
    n = len(arr)
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    # Traverse through all array elements
    for i in range(n - 1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n - i - 1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if float(arr[j][target_column]) > float(arr[j + 1][target_column]):
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return

def sorted_data(formatted_data,name_col,target_col):
    bubbleSort(formatted_data,target_col)
    sorted_data = []
    for entry in formatted_data:
        sorted_data.append({"name":entry[name_col],"value":entry[target_col]})
    return sorted_data

def make_display_values(formatted_data,name_col,target_col,multiplier,suffix):
    data = sorted_data(formatted_data,name_col,target_col)
    display_values = []
    for entry in data:
        desired_value = float(entry["value"]) * float(multiplier)
        display_values.append({"name": entry["name"], "value": f'{desired_value:,}' + " " + suffix})
    return display_values

def get_country_flag_path(country_name):
    return country_name + "_PATH_PLACEHOLDER"

def get_response_video_path(index,number_of_responses):
    if index == 0:
        return "FIRST_PATH" + str(random.randint(0,1000))
    elif index == number_of_responses - 1:
        return "LAST_PATH" + str(random.randint(0,1000))
    else:
        return "MID_PATH" + str(random.randint(0,1000))

def get_idle_face_video_path(index):
    return "IDLE_PATH"

def make_video_file_from_display_values(display_values,prompt_path):
    lines = []
    # make intro_line
    country_names = []
    for entry in display_values:
        country_names.append(entry["name"])

    #swap values so they are not in order
    temp = country_names[1]
    country_names[1] = country_names[2]
    country_names[2] = temp

    line = "INTRO"
    prompter_index = 2
    for index, country in enumerate(country_names):
        line += "," + country + ";" + get_country_flag_path(country)
        if prompter_index == index:
            line += ";" + prompt_path
        else:
            line += ";" + get_idle_face_video_path(index)
    lines.append(line)

    used_response_videos = []
    for index,entry in enumerate(display_values):
        line = "NORM"
        response_path = get_response_video_path(index,len(display_values))
        while response_path in used_response_videos:
            response_path = get_response_video_path(index, len(display_values))
        line += "," + response_path
        line += "," + "CAPTION"
        line += "," + entry["value"]
        if index == 0:
            line += ",NA"
        else:
            line += "," + display_values[index-1]["value"]
        line += "," + entry["name"]
        line += "," + get_country_flag_path(entry["name"])
        lines.append(line)
    return lines






format_data = get_first_row_label_data("csvData.csv",',')
n_random = get_n_random_entries(4,format_data)
display_values = make_display_values(n_random,"name","pop2022",1000,"people")
lines = make_video_file_from_display_values(display_values,"PROMPT_PATH")
for line in lines:
    print(line)
