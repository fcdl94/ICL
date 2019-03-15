import os


def save_results(file, acc_base, acc_new, acc_cum):
    print_header = False
    if not os.path.isfile(file):
        print_header = True
    with open(file, "a") as f:
        if print_header:
            f.write("acc_base,acc_new,acc_cum\n")
        f.write(f"{acc_base},{acc_new},{acc_cum}\n")


def create_log_folder(log):
    if not os.path.isdir(log):
        os.mkdir(log)
