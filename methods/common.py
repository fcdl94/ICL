import os


def save_results(file, acc_base, acc_new, acc_cum):
    print_header = False
    if not os.path.isfile(file):
        print_header = True
    with open(file, "a") as f:
        if print_header:
            f.write("acc_base,acc_new,acc_cum")
        f.write(f"{acc_base},{acc_new},{acc_cum}")
