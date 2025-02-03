import os
import pandas as pd


def convert_df_elements_to_float(df):
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


df_human = pd.read_csv("../Dataset/human-survey-results.csv").iloc[:, :-3]
df_human.columns = [col.strip('"').lower() for col in df_human.columns]


for file_name in os.listdir('./rq1'):
    df = pd.read_csv(f'./rq1/{file_name}', index_col=0)
    df = convert_df_elements_to_float(df)
    if 'Std' in df.columns or 'Avg' in df.columns:
        df.drop(columns=['Std', 'Avg'], inplace=True)
    df = df.loc[df_human.columns]

    if df.iloc[0, 0] >= 1:
        df = df[:]/100

    new_file_name = file_name.split(' - ')[-1]
    if 'long_context' in new_file_name:
        new_file_name = new_file_name.replace('long_context', 'ENC')
    if 'short_context' in new_file_name:
        new_file_name = new_file_name.replace('short_context', 'CNC')
    if 'gpt-3.5-turbo' in new_file_name:
        new_file_name = new_file_name.replace('gpt-3.5-turbo', 'GPT-3.5')
    if 'gpt-4' in new_file_name:
        new_file_name = new_file_name.replace('gpt-4', 'GPT-4')
    if 'LLaMa7B' in new_file_name:
        new_file_name = new_file_name.replace('LLaMa7B', 'Llama-2-7B')
    if 'LLaMa13B' in new_file_name:
        new_file_name = new_file_name.replace('LLaMa13B', 'Llama-2-13B')
    if 'chinese' in new_file_name:
        new_file_name = new_file_name.replace('chinese', 'Chinese')
    if 'baidu' in new_file_name:
        new_file_name = new_file_name.replace('baidu', 'ERNIE-4.0')
    if 'single-gender-statement(He)' in new_file_name:
        new_file_name = new_file_name.replace('single-gender-statement(He)', 'MCNC_')
    if 'single-gender-statement(She)' in new_file_name:
        new_file_name = new_file_name.replace('single-gender-statement(She)', 'FCNC_')

    assert df.shape[0] == 17
    if 'MCNC' in new_file_name or 'FCNC' in new_file_name:
        assert df.shape[1] == 10
    if 'ENC' in new_file_name:
        assert df.shape[1] == 11
    if 'CNC' in new_file_name and ('MCNC' not in new_file_name != 'FCNC' not in new_file_name):
        assert df.shape[1] == 15
    with open(f'./new_rq1/{new_file_name}', 'w') as f:
        f.write(df.to_csv(header=False, na_rep='NA'))

    print()