import os
import csv

path = "./extra/"

num_tables = 0
num_columns = 0
num_rows = 0

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            with open(os.path.join(root, file), "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = list(reader)
                num_tables += 1
                num_columns += len(rows[0]) if rows else 0
                num_rows += len(rows)

print(f"Total Tables: {num_tables}")
print(f"Total Columns: {num_columns}")
print(f"Total Rows: {num_rows}")
