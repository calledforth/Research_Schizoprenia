import csv

# Examine FNC data
print("Examining FNC data:")
with open("data/raw/mlsp_2014/Train/train_FNC.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"Number of columns: {len(header)}")
    print(f"First 10 column names: {header[:10]}")

    # Read first row
    first_row = next(reader)
    print(f"First row ID: {first_row[0]}")
    print(f"First 5 values: {first_row[1:6]}")

# Examine SBM data
print("\nExamining SBM data:")
with open("data/raw/mlsp_2014/Train/train_SBM.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"Number of columns: {len(header)}")
    print(f"First 10 column names: {header[:10]}")

    # Read first row
    first_row = next(reader)
    print(f"First row ID: {first_row[0]}")
    print(f"First 5 values: {first_row[1:6]}")

# Examine labels
print("\nExamining labels:")
with open("data/raw/mlsp_2014/Train/train_labels.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"Header: {header}")

    # Read first few rows
    for i, row in enumerate(reader):
        if i < 5:
            print(f"Row {i+1}: {row}")
        else:
            break
