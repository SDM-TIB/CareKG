from core.table_layer import TableLayer
from core.causal_layer import regression_analysis


def writerAge2FirstPublish():
    configs = {
    "endpoint_url": "http://localhost:7200/repositories/DBPediaW",
    "prefixes": {"dbo": "http://dbpedia.org/ontology/", "dbp": "http://dbpedia.org/resource/",
                 "ns2": 'http://xmlns.com/foaf/0.1/',
                 "dbpedia2": "http://dbpedia.org/property/", "xls": "http://www.w3.org/2001/XMLSchema#"},

    # (concept, concept brief name, concept type, prefix of concept)
    # concept type: 0: class, 1: property, 2: attribute
    # prefix: data type if concept is attribute else str
    "concepts": {0: ("University", 'univ', 0, 'dbo'), 1: ("Writer", 'writer', 0, 'dbo'), 2: ("Book", 'book', 0, 'dbo'),
                     3: ("arwuW", 'hasRank', 1, 'dbo'), 4: ("hasForStudent", 'isUniversityOf', 1, 'dbo'),
                     5: ("birthDate", 'hasBirth', 1, 'dbo'), 6: ("genre", 'hasGenre', 1, 'dbo'),
                     7: ("gender", 'hasGender', 1, 'ns2'), 8: ("author", 'authorOf', 1, 'dbo'),
                     9: ("releaseDate", 'releaseWhen', 1, 'dbo'), 10: ("Rank", 'rank', 2, int),
                     11: ("BirthDate", 'birth', 2, int), 12: ("Genre", 'genre', 2, str),
                     13: ("Gender", 'gender', 2, str), 20: ("ReleaseDate", 'bookDate', 2, int)},
    "causal_graph": [],
    # "graph_pattern": [[0, 3, 10], [0, 4, 1], [1, 5, 11], [1, 7, 13], [1, 8, 2], [2, 9, 14], [2, 6, 12]],  # can use id or brief concept
    "graph_pattern": [[1, 5, 11], [1, 8, 2], [2, 9, 20]],        # birth of author impact on publishdate
    "perspective": [1],
    "attr_design": None,
    "agg_strategy": {20: {'min': None},
                     11: {'min': None}
                     },
                     # 10: {'comb': {'comb_names': ['ALKorEGFR'], 'comb_vals': [['ALK', 'EGFR']], 'prior': False}},
                     # 0: {'comb': {'comb_names': ['math_prof'], 'comb_vals': [['http://www.example.com/prof3', 'http://www.example.com/prof4']], 'prior': False}}},
    "constraints": None
    }

    table_layer = TableLayer(configs=configs)
    # unit_df = table_layer.unit_table()

    causal_table, treatment, outcome = table_layer.causal_table(treatment=11,
                                                                outcome=20,
                                                                treatment_design=None,
                                                                # set() for treatment, or > threshold
                                                                # has_interference=True,
                                                                # peer_treatment_fun=np.mean,
                                                                # peer_threshold=204,
                                                                # id4peers=0,
                                                                # exclude_invalid=False
                                                                )
    causal_table.to_csv('tmp.csv')
    # print("treatment=", treatment, "outcome=", outcome)
    print(regression_analysis(causal_table, treatment, outcome))


def rankOfUniv2FirstPublish():
    configs = {
        "endpoint_url": "http://localhost:7200/repositories/DBPediaW",
        "prefixes": {"dbo": "http://dbpedia.org/ontology/", "dbp": "http://dbpedia.org/resource/",
                     "ns2": 'http://xmlns.com/foaf/0.1/',
                     "dbpedia2": "http://dbpedia.org/property/", "xls": "http://www.w3.org/2001/XMLSchema#"},
        # (concept, concept brief name, concept type, prefix of concept)
        # concept type: 0: class, 1: property, 2: attribute
        # prefix: data type if concept is attribute else str
        "concepts": {0: ("University", 'univ', 0, 'dbo'), 1: ("Writer", 'writer', 0, 'dbo'), 2: ("Book", 'book', 0, 'dbo'),
                     3: ("arwuW", 'hasRank', 1, 'dbo'), 4: ("hasForStudent", 'isUniversityOf', 1, 'dbo'),
                     5: ("birthDate", 'hasBirth', 1, 'dbo'), 6: ("genre", 'hasGenre', 1, 'dbo'),
                     7: ("gender", 'hasGender', 1, 'ns2'), 8: ("author", 'authorOf', 1, 'dbo'),
                     9: ("releaseDate", 'releaseWhen', 1, 'dbo'), 10: ("Rank", 'rank', 2, int),
                     11: ("BirthDate", 'birth', 2, int), 12: ("Genre", 'genre', 2, str),
                     13: ("Gender", 'gender', 2, str), 14: ("ReleaseDate", 'bookDate', 2, int)},
        "causal_graph": [],

        # "graph_pattern": [[0, 3, 10], [0, 4, 1], [1, 5, 11], [1, 7, 13], [1, 8, 2], [2, 9, 14], [2, 6, 12]],  # can use id or brief concept
        "graph_pattern": [[0, 3, 10], [0, 4, 1], [1, 8, 2], [2, 9, 14]],  # birth of author impact on publishdate
        "perspective": [1],
        "attr_design": None,
        "agg_strategy": {14: {'min': None},
                         # 11: {'min': None},
                         10: {'avg': None},
                         },
        # 10: {'comb': {'comb_names': ['ALKorEGFR'], 'comb_vals': [['ALK', 'EGFR']], 'prior': False}},
        # 0: {'comb': {'comb_names': ['math_prof'], 'comb_vals': [['http://www.example.com/prof3', 'http://www.example.com/prof4']], 'prior': False}}},
        "constraints": None
    }

    table_layer = TableLayer(configs=configs)
    # unit_df = table_layer.unit_table()

    causal_table, treatment, outcome = table_layer.causal_table(treatment=10,
                                                                outcome=14,
                                                                treatment_design=None,
                                                                # set() for treatment, or > threshold
                                                                # has_interference=True,
                                                                # peer_treatment_fun=np.mean,
                                                                # peer_threshold=204,
                                                                # id4peers=0,
                                                                # exclude_invalid=False
                                                                )
    causal_table.to_csv('tmp.csv')
    # print("treatment=", treatment, "outcome=", outcome)
    print(regression_analysis(causal_table, treatment, outcome))


# TODO Exp1: Birthe Date of Author -> First Publish Year of Author
writerAge2FirstPublish()

# TODO Exp2: Rank of University of Author -> First Publish Year of Author
# rankOfUniv2FirstPublish()

