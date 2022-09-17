from core.table_layer import TableLayer
from core.causal_layer import regression_analysis, avg_cause_effect


def try_movie_example(setting):
    '''
    :param setting: 0: Director, 1:Director and Actor, 2: Director not Actor
    :return:
    '''
    configs = {
        "endpoint_url": "http://localhost:7200/repositories/movie",
        "prefixes": {"ex": "http://www.example.com/", "xls": "http://www.w3.org/2001/XMLSchema#"},
        "concepts": {0: "Movie", 1: "Person", 2: "Actor", 3: "Director", 4: "Recruit", 5: 'hasScucess',
                     6: "hasFame", 7: 'hasExp', 8: "actIn", 9: 'direct', 10: "hasActor", 11: 'hasDirector',
                     12: "hasMovie", 13: 'experience', 14: 'success', 15: 'fame'},
        "concepts_type": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],     # 0:class, 1:property, 2:attribute
        "concepts_brief": ["move", 'person', 'actor', 'director', 'recruit', 'hassuc', 'hasfame', 'hasexp',
                           'actin', 'direct', 'hasactor', 'hasdirector', 'hasmovie', 'exp', 'success', 'fame'],  # variable names ?prof, ?univ
        "concepts_prefix": ["ex", "ex", "ex", "ex", "ex", "ex", "ex", "ex", "ex", "ex", "ex", "ex", "ex", int, int, int],
        # according to the correspond index; type means attribute
        "graph_pattern": [[0, 5, 14], [3, 7, 13], [3, 9, 0]], #, [2, 6, 15], [2, 8, 0],  [4, 10, 2], [4, 11, 3],
                          # [4, 12, 0]],  # can use id or brief concept
        # don't choose both path, because there will less instance to meet the requirement
        "perspective": [3],
        "attr_design": None,
        "agg_strategy": {14: {'avg': None},
                         15: {'avg': None}},
                         # 15: {'repl': {'old_vals': [0, 1], 'new_vals':[1, 2]}}},
        "constraints": {3: {  # constraint for student  "min_card": {"path": [[2, 6, 8]], "card": [2]},
            "class": {"is": [2, 3] if setting == 1 else [3], "not": [] if setting == 1 else [2]}}
        }
    }
    table_layer = TableLayer(configs=configs)
    # unit_df = table_layer.unit_table()
    # unit_df.to_csv('tmp.csv')
    causal_table, treatment, outcome = table_layer.causal_table(treatment=13,
                                                                outcome=14,
                                                                treatment_design=0.5,          # set() for treatment, or > threshold
                                                                # has_interference=True,
                                                                # peer_treatment_fun=np.mean,
                                                                # peer_threshold=204,
                                                                # id4peers=0,
                                                                exclude_invalid=False if setting==0 else True)
    causal_table.to_csv('tmp.csv')
    # causal_table.drop(columns=['avg_fame'], inplace=True)
    causal_table['success'] = causal_table['success'].apply(lambda x: 0 if x < 0.5 else 1)
    print(causal_table.columns)

    print(avg_cause_effect(causal_table, treatment=treatment, Y=outcome))


# TODO Exp1
# TODO Relational Path: Director.Direct.Movie ;
# TODO Causality: Director's Experience -> Movie's Success ;
# TODO Constraint: No
# try_movie_example(0)


# TODO Exp2
# TODO Relational Path: Director.Direct.Movie ;
# TODO Causality: Director's Experience -> Movie's Success ;
# TODO Constraint: those Director who are also Actor
try_movie_example(1)


# TODO Exp3
# TODO Relational Path: Director.Direct.Movie ;
# TODO Causality: Director's Experience -> Movie's Success ;
# TODO Constraint: those Director who are not Actor
# try_movie_example(2)

clarify: PatientShape
    a sh:NodeShape ;
    Sh: targetClass: clarify:Patient.