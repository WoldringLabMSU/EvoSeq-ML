import pandas as pd

# Load the dataset
df = pd.read_csv('Supports.csv')  # Replace 'path_to_your_file.csv' with the actual path

# List 1: 2nd column values > 95
# Assuming the '2nd column' means index 1 (as pandas indexing starts from 0)
list_1 = df[df.iloc[:, 1] > 95]

# List 2: Last column values > 80
# Assuming 'last column' can be accessed with -1 index
list_2 = df[df.iloc[:, -1] > 80]

# List 3: 2nd column > 0.95 AND last column > 80
# Adjusted the criteria for 2nd column to > 0.95 as per your request, assuming a typo in the first criteria
list_3 = df[(df.iloc[:, 1] > 95) & (df.iloc[:, -1] > 80)]


# Saving the lists to files or processing further as required
list_1.to_csv('ultra_only.csv')
list_2.to_csv('sh_only.csv')
list_3.to_csv('ultra_sh.csv')

# awk -F, 'NR==1{for (i=1; i<=NF; i++) if ($i == "Node") c=i} c{print $c}' Endo_sh_only.csv