from data_layer import ConceptMapping, QueryBuilder, QueryEngine
from data_layer import json2df
from functools import partial, reduce
import operator
from pandas import DataFrame
from collections import Counter
import numpy as np
from causal_layer import avg_cause_effect, regression_analysis, relational_cause_effect_xlearner, relational_cause_effect_tlearner, relational_cause_effect_slearner


def configs_check(configs: dict):
    concepts = configs['concepts']
    causal_graph = configs['causal_graph']
    graph_pattern = configs['graph_pattern']
    perspective = configs['perspective']
    attr_design = configs['attr_design']
    agg_strategy = configs['agg_strategy']

    classes = set()
    properties = set()
    attributes = set()
    for index_, quadruple in concepts.items():
        assert type(quadruple[0]) is str and type(quadruple[1]) is str and quadruple[2] in [0, 1, 2]
        if quadruple[2] == 0:
            classes.add(index_)
        elif quadruple[2] == 2:
            assert type(quadruple[3]) is type
            attributes.add(index_)
        else:
            assert type(quadruple[3]) is str
            properties.add(index_)
    if causal_graph:
        for c, e in causal_graph:
            assert c in attributes and e in attributes

    for sub, predicate, obj in graph_pattern:
        assert sub in classes
        assert predicate in properties
        assert obj in classes or obj in attributes

    for concept in perspective:
        assert concept in classes

    if attr_design:
        for k in attr_design.keys():
            assert k not in properties

    for k in agg_strategy.keys():
        assert k not in properties


class TableLayer:

    def __init__(self, configs):
        self.index_map = {}
        self.configs = self.reindex_concepts(configs)
        configs_check(self.configs)

        self.sparql_engine = QueryEngine(endpoint_url=configs['endpoint_url'])

        concepts_size = len(configs['concepts'].keys())
        concepts = [0] * concepts_size
        concepts_brief = [0] * concepts_size
        concepts_type = [0] * concepts_size
        concepts_prefix = [0] * concepts_size

        for i in range(concepts_size):
            concept, brief, concept_type, prefix = configs['concepts'][i]
            concepts[i] = "<" + configs['prefixes'][prefix] + concept + ">" if type(prefix) is str else concept if (not prefix) else prefix
            concepts_brief[i] = brief
            concepts_type[i] = concept_type
            concepts_prefix[i] = prefix
        
        # concepts = ["<"+prefix+concept+">" if type(prefix) is str else concept if prefix is None else prefix for prefix, concept in zip([configs['prefixes'][e] if type(e) is str else e for e in configs['concepts_prefix']], concepts)]

        self.map = ConceptMapping(concepts_brief, concepts, concepts_type)
        self.querybuilder = QueryBuilder(query_engine=self.sparql_engine,
                                         concept_mapping=self.map,
                                         graph_structure=configs['graph_pattern'],
                                         prefix={full: brief for brief, full in configs['prefixes'].items()})

        self.constraint_queries = [self.querybuilder.constraint_query(target_class=identity, constraints=constraint) for identity, constraint in configs["constraints"].items()] if configs["constraints"] else None

        if self.constraint_queries:
            for qry in self.constraint_queries:
                print("constraint query:\n"+qry+'\n\n')

        self.data_query_builder = partial(self.querybuilder.sparql_query,
                                          perspective=configs['perspective'],
                                          attr_design=configs["attr_design"],
                                          agg_strategy=configs["agg_strategy"])

    def reindex_concepts(self, configs: dict):
        '''
        :param configs:
        :return:
        # [(concept, brief, type, prefix/datatype)], type: 0:class, 1:property, 2:attribute

        '''
        concepts = configs['concepts']
        causal_graph = configs['causal_graph']
        graph_pattern = configs['graph_pattern']
        perspective = configs['perspective']
        attr_design = configs['attr_design']
        agg_strategy = configs['agg_strategy']
        constraints = configs['constraints']

        self.index_map = {i: _i for _i, i in enumerate(concepts.keys())}
        configs['concepts'] = {self.index_map[i]: c for i, c in concepts.items()}
        if causal_graph:
            configs['causal_graph'] = [(self.index_map[c], self.index_map[e]) for c, e in causal_graph]
        configs['graph_pattern'] = [[self.index_map[i] for i in triple] for triple in graph_pattern]
        configs['perspective'] = [self.index_map[i] for i in perspective]
        if attr_design:
            configs['attr_design'] = {self.index_map[i]: design for i, design in attr_design.items()}
        configs['agg_strategy'] = {self.index_map[i]: agg for i, agg in agg_strategy.items()}
        if constraints:
            configs['constraints'] = {self.index_map[i]: constraint for i, constraint in constraints.items()}
        return configs

    def convert_index(self, *args):
        return (self.index_map[arg] for arg in args)

    @property
    def invalid_entities(self):
        if not self.constraint_queries:
            return None

        def exec_one(qry_):
            df = json2df(self.sparql_engine.query(query=qry_))
            return df.columns[0], list(df[df.columns[0]])
        results = {}
        for qry in self.constraint_queries:
            ent_name, entities = exec_one(qry)
            results[ent_name] = entities
        return results

    def unit_table(self, include_peers: bool = False, id4peers=None, exclude_invalid: bool = False):
        '''
        :param include_peers: whether consider peers
        :param id4peers: which identifier considered to define the peers who share same entities (identifiers)
        :param exclude_invalid: whether exclude invalid entities?
        :return:
        '''
        assert bool(include_peers) == bool(id4peers is not None)
        # print(self.invalid_entities)
        # input()
        qry = self.data_query_builder(invalid_entities=self.invalid_entities if exclude_invalid else None)
        print("data query:\n"+qry+"\n\n")
        df = json2df(self.sparql_engine.query(qry))
        df = self.calculate_peers(df, id4peers) if include_peers else df

        # remove all identifiers except those identifiers which used as attributes
        drop_cols = [id_col for id_col in self.map.classes if id_col in df.columns and id_col not in self.querybuilder.identifier_as_att]
        df.to_csv("tmp2.csv")
        return df.drop(columns=drop_cols)   # , df

    def calculate_peers(self, df: DataFrame, id4peers=None):
        if id4peers is not None:
            id4peers = self.map.id2name(id4peers) if type(id4peers) is int else id4peers
        else:
            return df
        assert id4peers in self.map.classes

        id_col = id4peers
        # concept = self.map.name2uri(id_col)
        # prefix = concept[1:concept.rfind('/')+1]
        #
        # prefix = None if prefix not in self.querybuilder.prefix else prefix
        # if prefix:
        #     set_col = [set([s.replace(prefix, '') for s in x.split(';')]) for x in df[id_col]]
        # else:
        # todo remove prefix
        set_col = [set([e[e.rfind['/']+1:] if '/' in e else e for e in x.split(';')]) for x in df[id_col]]

        peers = [set() for _ in range(len(df))]
        for _id, freq in Counter(reduce(operator.concat, [list(s) for s in set_col])).items():
            if freq == 1:
                continue
            id_val_peers = set([i for i, id_s in enumerate(set_col) if _id in id_s])
            for p in id_val_peers:
                peers[p] = peers[p].union(id_val_peers)
        peers = [list(peer_set.difference(set([i]))) for i, peer_set in enumerate(peers)]
        df['peers'] = peers

        return df

    def causal_table(self, treatment,
                     outcome,
                     treatment_design=None,  # set() for treatment, or > threshold, or None for keep as continuous treatment
                     has_interference=False,
                     peer_treatment_fun=None,
                     peer_threshold=0.5,
                     # include_peers=False,
                     id4peers=None,
                     exclude_invalid=False):
        '''
        :param treatment: brief name of attribute or index of attribute. e.g. 'rank' or 7
        :param outcome: brief name of attribute or index of attribute. e.g. 'score' or 8
        :param treatment_design:
        1. set([val, val, ...]) for treatment group;
        2. threshold. control group(0) if value < threshold else treatment group(1)
        3. None for continuous treatment
        :param has_interference: whether consider interference?
        :param peer_treatment_fun: what kind of aggregation function used for peer treatment after treatment processed? sum, mean, median
        :param peer_threshold: control group(0) if aggregated value < peer_threshold else treatment group(1)
        :param id4peers: which identifier considered to define the peers who share same entities (identifiers)
        :param exclude_invalid: whether exclude invalid entities?
        :return:
        '''
        treatment, outcome = self.convert_index(treatment, outcome)

        unit_df = self.unit_table(has_interference, id4peers, exclude_invalid)
        treatment = self.map.id2name(treatment) if type(treatment) is int else treatment
        outcome = self.map.id2name(outcome) if type(outcome) is int else outcome
        assert set([treatment, outcome]).issubset(self.map.attributes)   # treatment and outcome should be att
        assert has_interference == bool(peer_treatment_fun) == bool(id4peers is not None)

        def rename_complex_name(col):
            if col not in unit_df.columns:
                for c in unit_df.columns:
                    if "_"+col in c:
                        unit_df.rename(columns={c: col}, inplace=True)
                        return

        rename_complex_name(treatment)
        rename_complex_name(outcome)

        treatment_type = type(unit_df[treatment][0])
        if treatment_type is str:
            assert type(treatment_design) is set
            unit_df[treatment] = unit_df[treatment].apply(lambda x: 1 if x in treatment_design else 0)
        elif treatment_type is None:
            pass    # continuous variable
        elif treatment_type not in [int, float, np.int64, np.float64]:
            raise Exception(type(unit_df[treatment][0]), "type of treatment is not supported by CareKG!")

        # only numerical treatment can pass here
        # unit_df = pandas.DataFrame()
        if has_interference:
            len_before = len(unit_df)
            unit_df = unit_df.loc[(unit_df['peers'].str.len() != 0)]
            print("TableLayer.causal_table(): ", len_before - len(unit_df), "rows are removed due to no peers")

            unit_df[treatment+'_peers'] = unit_df['peers'].apply(lambda x: 0 if np.nan_to_num(peer_treatment_fun(unit_df[treatment].iloc[x])) < peer_threshold else 1)
            unit_df.rename(columns={treatment: treatment+"_ego"}, inplace=True)

        # discretize treatment into 0, 1
        treatment1 = treatment+"_ego" if has_interference else treatment
        if type(treatment_design) in [int, float]:
            unit_df[treatment1] = unit_df[treatment1].apply(lambda x: 0 if x < treatment_design else 1)
        elif type(treatment_design) is set:
            unit_df[treatment1] = unit_df[treatment1].apply(lambda x: 1 if x in treatment_design else 0)
        elif treatment_design is None:
            if has_interference:    # keep
                raise Exception(treatment, "must be designed as 0 or 1 for peer effect analysis")
            else:
                pass
        else:
            raise Exception("treatment_design should be int, float, or set")

        # remove peers column
        if 'peers' in unit_df.columns:
            unit_df.drop(columns=['peers'], inplace=True)

        # digital other features
        for col in unit_df.columns:
            if type(unit_df[col][0]) in [int, float, np.int64, np.float64]:
                continue
            elif type(unit_df[col][0]) is not str:
                raise Exception("cannot process the datatype of", type(unit_df[col][0]))

            uni_val_dict = {val: i+1 for i, val in enumerate(unit_df[col].unique())}
            if len(uni_val_dict) > 20:
                print("pay attention:", col, "has more than 20 unique values!")
            unit_df[col].replace(to_replace=uni_val_dict, inplace=True)
            print("auto replace categorical variable: ", col, uni_val_dict)
        return unit_df, treatment, outcome


def toy_toy_example():
    configs = {
    "endpoint_url": "http://localhost:7200/repositories/university",
    "prefixes": {"ex": "http://www.example.com/", "xls": "http://www.w3.org/2001/XMLSchema#"},
    "concepts": {0: ("Professor", "prof", "ex"), 1: ("University", "univ", "ex"), 2: ("Student", "stu", "ex"),
                 3: ("hasRank", 'hasR', 'ex'), 4: ("workIn", 'workIn', 'ex'), 5: ("supervise", 'supervise', 'ex'),
                 6: ("hasScore", 'hasS', 'ex'), 7: ("rank", 'rank', int), 8: ("score", 'score', float),
                 9: ("hasMutation", 'hasM', 'ex'), 10: ("mutation", 'mutation', str)},
    # "concepts_brief": ["prof", "univ", "stu", "hasR", "workIn",
    #                    "supervise", "hasS", "rank", "score", "hasM", "mutation"],   # variable names ?prof, ?univ
    # "concepts_prefix": ["ex", "ex", "ex", "ex", "ex", "ex", "ex", int, float, "ex", str],    # according to the correspond index; type means attribute
    "graph_pattern": [[0, 4, 1], [1, 3, 7], [0, 5, 2], [2, 6, 8], [2, 9, 10]],  # can use id or brief concept
    "perspective": [1, 2],
    "attr_design": {8: {'avg': None},
                    10: {'comb': {'comb_names': ['ALKorEGFR', "No"], 'comb_vals': [['ALK', 'EGFR'], ['']], 'prior': False}}},
    "agg_strategy": {8: {'max': None},
                     10: {'comb': {'comb_names': ['ALKorEGFR'], 'comb_vals': [['ALK', 'EGFR']], 'prior': False}},
                     0: {'comb': {'comb_names': ['math_prof'], 'comb_vals': [['http://www.example.com/prof3', 'http://www.example.com/prof4']], 'prior': False}}},
    "constraints": {2: {     # constraint for student  "min_card": {"path": [[2, 6, 8]], "card": [2]},
                        "class": {"is": [2], "not": [0]}
                        }
                    }
    }
    # table_layer = TableLayer(configs=configs)
    # causal_df, unit_df = table_layer.unit_table()
    # causal_df, unit_df = table_layer.unit_table(exclude_invalid=True)
    # causal_df, unit_df = table_layer.unit_table(include_peers=True, id4peers=2)
    # unit_df = table_layer.unit_table(exclude_invalid=True, include_peers=True, id4peers=2)
    # unit_df.to_csv('tmp1.csv')

    # causal_table, treatment, outcome = table_layer.causal_table(treatment=7,
    #                      outcome=8,
    #                      treatment_design=204,          # set() for treatment, or > threshold
    #                      has_interference=True,
    #                      peer_treatment_fun=np.mean,
    #                      peer_threshold=204,
    #                      id4peers=0,
    #                      exclude_invalid=False)

    # causal_table.to_csv('tmp.csv')
    #
    # from causal_layer import relational_cause_effect_xlearner
    #
    # print(relational_cause_effect_xlearner(causal_table, treatment, outcome))



def try_lung_cancer():
    prefix = "http://clarify2020.eu/vocab/"
    configs = {
    "endpoint_url": "https://labs.tib.eu/sdm/clarify-kg-8-0/sparql",
    "prefixes": {"clf": "http://clarify2020.eu/vocab/", "xls": "http://www.w3.org/2001/XMLSchema#"},
    "concepts": {0:'LCPatient', 1:'has_LC_SLCG_ID', 2:'age', 3:'sex', 4:'hasFamilyHistory', 5:'hasFamilyCancerType',
                 6: 'hasBio', 7: None, 8: 'Age', 9: 'Sex', 10: None, 11: 'cancer', 12: 'mutation'},
    "concepts_brief": ['LCpatient', 'has_id', 'hasage', 'hassex', 'hasfamily', 'hascancer', 'hasmutation',
                       'patientId', 'age', 'sex', 'family', 'cancer', 'mutation'],   # variable names ?prof, ?univ
    "concepts_prefix": ['clf','clf','clf','clf','clf','clf','clf', None, int, str, None, str, str],    # according to the correspond index; type means attribute
    "graph_pattern": [[0, 1, 7], [7, 4, 10], [10, 5, 11], [7, 2, 8], [7, 3, 9], [7, 6, 12]],  # can use id or brief concept
    "perspective": [7],
    "attr_design": None,
    "agg_strategy": {11: {'comb': {'comb_names': ['MajorCancer'], 'comb_vals': [[prefix+'Breast', prefix+'Lung',
                                                                          prefix+'Colorrectal', prefix+'Head_and_neck',
                                                                          prefix+'Uterus/cervical', prefix+'Esophagogastric',
                                                                          prefix+'Prostate', prefix+'Leukemia']], 'prior': False}},
                     12: {'comb': {'comb_names': ['ALKorEGFR'], 'comb_vals': [[prefix+'ALK', prefix+'EGFR']], 'prior': False}}
                     },
                     # 10: {'comb': {'comb_names': ['ALKorEGFR'], 'comb_vals': [['ALK', 'EGFR']], 'prior': False}},
                     # 0: {'comb': {'comb_names': ['math_prof'], 'comb_vals': [['http://www.example.com/prof3', 'http://www.example.com/prof4']], 'prior': False}}},
    "constraints": None
    }

    table_layer = TableLayer(configs=configs)
    unit_df = table_layer.unit_table()
    unit_df.to_csv('tmp.csv')




