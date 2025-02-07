import sys
if len(sys.argv) != 4:
    print("Usage: python script.py file1 file2 file3")
    sys.exit(1)
file1_name = sys.argv[1]
file2_name = sys.argv[2]
file3_name = sys.argv[3]
file1_dict = {}
with open(file1_name, 'r') as f1:
    for line in f1:
        fields = line.strip().split('\t')
        if len(fields) >= 4:
            start, end, depth = int(fields[1]), int(fields[2]), float(fields[3])
            for i in range(start, end):
                file1_dict[(fields[0], i)] = depth
with open(file2_name, 'r') as f2, open(file3_name, 'w') as f3:
    for line in f2:
        fields = line.strip().split('\t')
        if len(fields) >= 3:
            chromosome, start, end = fields[0], int(fields[1]), int(fields[2])
            for i in range(start, end):
                depth = file1_dict.get((chromosome, i), 0)
                f3.write(f"{chromosome}\t{i}\t{i+1}\t{depth}\n")
