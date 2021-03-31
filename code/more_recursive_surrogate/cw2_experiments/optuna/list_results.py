from sys import exit

import optuna
from optuna_utils import optuna_table

storage = "sqlite:///test.db"
summaries = optuna.study.get_all_study_summaries(storage)

while(True):
    print("List of all study names:")
    for i, s in enumerate(summaries):
        print("{:02d} {}".format(i, s.study_name))

    idx = int(input("Type number to show best trials: "))
    if 0 <= idx < len(summaries):
        name = summaries[idx].study_name
        optuna_table.print_table(name, "sqlite:///test.db")
    else:
        print("invalid index")

