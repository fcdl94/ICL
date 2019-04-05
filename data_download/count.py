import os
for dir in sorted(os.listdir(".")):
	count = 0                                                              
	for fi in os.listdir(f"./{dir}"):
		count += 1
	print(dir, str(count))

