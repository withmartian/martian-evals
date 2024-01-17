import os
import glob
import json
import pandas as pd


def find_jsonl_files(folder_path):
    """
    Traverse the directory and subdirectories for .jsonl files.
    Return a list of paths to these files.
    """
    return glob.glob(os.path.join(folder_path, '**/*.jsonl'), recursive=True)


def parse_jsonl_file(file_path):
    """
    Open and read the .jsonl file line by line.
    For each line, parse the JSON content.
    If a line contains the "spec" key, extract "completion_fns" and "eval_name".
    Return the extracted information.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                if 'spec' in data:
                    spec = data['spec']
                    return spec.get('completion_fns', []), spec.get('eval_name', None)
            except json.JSONDecodeError:
                continue
    return [], None


def create_dataframe(data):
    """
    Use the aggregated data to create a pandas DataFrame.
    Rows are "eval_name", columns are unique "completion_fns".
    Cell values indicate the presence of a corresponding .jsonl file.
    """
    # Create an empty DataFrame with unique model names as columns
    model_names = sorted(set(model for models in data.values() for model in models))
    df_list = []

    # Fill the DataFrame
    for eval_name, models in data.items():
        df_row = {model: (model in models) for model in model_names}
        df_list.append(pd.Series(df_row, name=eval_name))

    return pd.DataFrame(df_list, columns=model_names)


def main(folder_path):
    """
    Main function to execute the script.
    Find all .jsonl files, parse each file, and aggregate data.
    Create and return the final DataFrame.
    """
    data = {}
    for file_path in find_jsonl_files(folder_path):
        completion_fns, eval_name = parse_jsonl_file(file_path)
        if eval_name:
            data[eval_name] = data.get(eval_name, []) + completion_fns

    return create_dataframe(data)


def generate_router_inputs_analysis(file_path: str):
    """
    Generate a DataFrame containing the information about the .jsonl files.
    """
    inputs_df = pd.read_pickle(file_path)
    result_df = inputs_df.pivot_table(index='eval_name', columns='model_name', aggfunc='size', fill_value=0)
    return result_df


if __name__ == '__main__':
    # Example usage (replace with the actual folder path)
    folder_path = '~/Desktop/ICML_router_benchmark_paper_data_records'
    expanded_user_path = os.path.expanduser(folder_path)
    df = main(expanded_user_path)
    # print(df)
    df.to_csv('~/Desktop/ICML_router_benchmark_paper_data_records.csv')