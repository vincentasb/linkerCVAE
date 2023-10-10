import csv
#Script for slicing full training data into chunks for debugging
def slice_csv(input_file, output_file, limit=10000):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            if i >= limit:
                break
            writer.writerow(row)

if __name__ == "__main__":
    input_file = "shuffled_csv_file.csv"
    output_file = "10knormalized_shuffled.csv"
    slice_csv(input_file, output_file)
