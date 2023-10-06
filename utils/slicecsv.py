import csv

def slice_csv(input_file, output_file, limit=30000):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            if i >= limit:
                break
            writer.writerow(row)

if __name__ == "__main__":
    input_file = "normalized_file.csv"
    output_file = "30knormalized.csv"
    slice_csv(input_file, output_file)
