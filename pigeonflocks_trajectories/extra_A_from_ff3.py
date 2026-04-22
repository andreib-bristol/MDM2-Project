import Trimmed_Flight_Data.ff3_data as ff3

# Step 1: get the dictionary
data = ff3.data

# Step 2: remove key "A"
data.pop("A", None)

# Step 3: write new file
with open("ff3_data_cleaned.py", "w") as f:
    f.write("data = {\n")
    for k, v in data.items():
        f.write(f"    {repr(k)}: {repr(v)},\n")
    f.write("}\n")
