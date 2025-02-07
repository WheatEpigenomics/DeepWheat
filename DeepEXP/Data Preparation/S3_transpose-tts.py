import sys

if len(sys.argv) != 3:
    print("Usage: python 123.py input_file output_file")
else:
    input_file = sys.argv[1]
    output_file = sys.argv[2]

  
    columns_per_row = 4000


    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        data = infile.read().splitlines() 
        rows = len(data) // columns_per_row  

        
        for i in range(rows):
            start = i * columns_per_row
            end = start + columns_per_row
            row_data = data[start:end]
            outfile.write("\t".join(row_data) + "\n")

    print(f"{input_file} turn {output_file}")
