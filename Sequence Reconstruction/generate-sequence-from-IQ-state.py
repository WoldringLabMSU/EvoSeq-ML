import sys
import pandas as pd
import numpy as np
import random
import subprocess

def stream_sequences_to_file(input_file, output_file, nodes, num_sequences=10000):
    df = pd.read_csv(input_file, delim_whitespace=True)
    df.columns = [col.replace('p_', '') for col in df.columns]

    output_sequences = {}  # Initialize the dictionary
    for node in nodes:
        node_df = df[df['Node'] == node]  # Filter the dataframe for the current node
        sequences = [generate_sequence(node_df) for _ in range(num_sequences)]
        output_sequences[node] = sequences

    with open(output_file, 'w') as f:
        for node, sequences in output_sequences.items():
            for sequence in sequences:
                f.write(">{0}\n{1}\n".format(node, sequence))

def generate_sequence(node_df):
    threshold = 0.2
    sequence = ''
    node_df = node_df.iloc[:,3:]
    for _, row_data in node_df.iterrows():
        numeric_row_data = row_data[pd.to_numeric(row_data, errors='coerce').notnull()]
        if not numeric_row_data.empty:
            max_likelihood = numeric_row_data.max()
            new_threshold = max_likelihood - 0.1 if max_likelihood < threshold else threshold
            valid_amino_acids = numeric_row_data[numeric_row_data >= new_threshold].index.tolist()
            sequence += random.choice(valid_amino_acids)
        else:
            sequence += '-'  # Placeholder
    return sequence

def insert_gaps(input_file, sequences_file, output_file):
    df = pd.read_csv(input_file, delim_whitespace=True)
    gap_positions = df[df['p_0'] > df['p_1']][['Node', 'Site']].groupby('Node')['Site'].apply(list).to_dict()

    with open(sequences_file, 'r') as f, open(output_file, 'w') as out_f:
        for seq_block in f.read().split('>')[1:]:
            node, sequence = process_sequence_block(seq_block, gap_positions)
            out_f.write(">{0}\n{1}\n".format(node, ''.join(sequence)))

def process_sequence_block(seq_block, gap_positions):
    lines = seq_block.strip().split('\n')
    node = lines[0]
    sequence = list(lines[1])

    for pos in gap_positions.get(node, []):
        if pos - 1 < len(sequence):
            sequence[pos - 1] = '-'
    return node, sequence

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python script.py input_file output_file num_sequences nodes_file gap_info_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_sequences = int(sys.argv[3])
    nodes_file = sys.argv[4]  # The path to a file containing node identifiers
    gap_info_file = sys.argv[5]  # The path to the gap information file

    # Read nodes from the file
    with open(nodes_file, 'r') as file:
        nodes = [line.strip() for line in file.readlines()]

    stream_sequences_to_file(input_file, output_file, nodes, num_sequences)

    # Assuming gap_info_file is correctly specified in the command line arguments
    final_output_with_gaps = output_file.replace('.fasta', '_with_gaps.fasta')
    insert_gaps(gap_info_file, output_file, final_output_with_gaps)
