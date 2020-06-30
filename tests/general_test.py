# from initial_setup import *
# import matplotlib.pyplot as plt
#
# def test_create_folder():
#     create_img_folder()
#
# def test_save_img_plot():
#     plt.plot([0, 1, 2, 3, 4], [0, 4, 5, 8, 12])
#     plt.savefig('img/test_plot.png')
#

#
#
# def test_string_dataframe():
#     df = pd.DataFrame({'mixed_types': [12331, '345', 'text']})
#
#     print(df)
#
#     print("df without string")
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df = df.dropna()
#     print(df)


#
# def test_drop_column():
#     # create a dictionary with five fields each
#     data = {
#         'A': ['A1', 'A2', 'A3', 'A4', 'A5'],
#         'B': ['B1', 'B2', 'B3', 'B4', 'B5'],
#         'C': ['C1', 'C2', 'C3', 'C4', 'C5'],
#         'D': ['D1', 'D2', 'D3', 'D4', 'D5'],
#         'E': ['E1', 'E2', 'E3', 'E4', 'E5']}
#
#     # Convert the dictionary into DataFrame
#     df = pd.DataFrame(data)
#
#     # Remove three columns as index base
#     df.drop(df.columns[[0, 4, 2]], axis=1, inplace=True)
#
#     print(df)

# import pandas as pd
#
# def testx():
#     df = pd.DataFrame({"enhancers": [[1, 2], [2, 2]], "promoters": [[3, 4], [6, 7]]})
#
#     df = df.to_numpy()
#
#     print(df)

# import compress_json
#
# def testxxxy():
#     compress_json.local_dump("results", "img/results_promoters.json")
#
#
# import json
#
# from progettobioinf.results import *
# import matplotlib.pyplot as plt
#
# def test_json():
#     with open('results2.json') as json_file:
#         results = json.load(json_file)
#         df_results = convert_results_to_dataframe(results)
#         df_results.to_csv('results2.csv', sep=',')
#
#     df_results = pd.read_csv('results.csv', sep=',')
#     df_results.groupby('model').plot.bar()

# from glob import glob
#
# import pandas as pd
# from PIL import Image
# from barplots import barplots
#
#
# def test_result():
#     df = pd.read_csv("results.csv")
#     print(df)
#
#     barplots(
#         df,
#         groupby=["model", "run_type"],
#         orientation="horizontal",
#         show_legend=False
#     )
#
#     for x in glob("barplots/*.png"):
#         Image.open(x)
