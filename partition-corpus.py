file_number = 1
file_counter = 0

files = []

with open("wikipedia2015-abstracts.txt") as f:
	for line in f:
		if len(line) < 10:
			continue

		files.append(line.strip())
		file_counter += 1

		if file_counter == 500000:				
			filename = "raw/reference-corpus/wiki2015-%s.txt" % file_number
			with open(filename, 'w') as f:
    				for file in files:
        				f.write("%s\n" % file)
			print('Written %s files' % file_number)
			files = []
			file_counter = 0
			file_number += 1

	filename = "raw/reference-corpus/wiki2015-%s.txt" % file_number
	with open(filename, 'w') as f:
		for file in files:
			f.write("%s\n" % file)			


