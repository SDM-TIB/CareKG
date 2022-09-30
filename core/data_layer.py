from SPARQLWrapper import SPARQLWrapper, JSON, RDF
import pandas
from typing import List, Tuple, Any
from functools import reduce
import operator

# TODO
# 1. casual structure to recognize the adapting set
# 2. support NULL value of other variables
# 3. context (strata) is not support yet

# package:
# pandas 1.3.0
# scikit-learn 1.0.2

def str2num(s):
    if type(s) is not str:
        return s
    if (s.find('-') <= 0) and s.replace('-', '', 1).isdigit():
        return int(s)
    elif (s.find('-') <= 0) and (s.count('.') < 2) and \
            (s.replace('-', '', 1).replace('.', '', 1).isdigit()):
        return float(s)
    else:
        return s


def json2df(json_ret):
    '''
    :param json_ret: json result from SPARQLWrapper
    :return: pandas.DataFrame
    '''
    # columns = json_ret['head']['vars']
    data = [{key: str2num(val['value']) for key, val in binding.items()} for binding in json_ret['results']['bindings']]
    return pandas.DataFrame(data)


def replace_clause(s: str, p: str, o1: str, o2: str, old_vals: list, new_vals: list, fun_name=''):
    '''
    :param s:
    :param p:
    :param o1:
    :param o2:
    :param old_vals:
    :param new_vals:
    :param fun_name: fun_name != '' when agg use this function
    :return:
    '''
    assert len(old_vals) == len(new_vals) and len(old_vals) > 0

    clause = 'replace(str({o2}), "{old}", "{new}")'.format(o2=o2, old=old_vals[0], new=new_vals[0])
    for old_val, new_val in zip(old_vals[1:], new_vals[1:]):
        clause = 'replace({o2}, "{old}", "{new}")'.format(o2=clause, old=old_val, new=new_val)
    o1 = "?"+fun_name+"_"+o1[1:] if fun_name else o1      # agg: agg_design_attribute for tracking functions
    return [' '.join([s, p, o2]) + '.', 'BIND({o2} AS {o1}).'.format(o2=clause, o1=o1)], o1


def discretize_clause(s: str, p: str, o1: str, o2: str, bins: list, vals: list, fun_name=''):  # [low, high)
    '''
    :param s:
    :param p:
    :param o1:
    :param o2:
    :param bins:
    :param vals:
    :param fun_name: fun_name != '' when agg use this function
    :return:
    '''
    assert len(bins) == len(vals) - 1
    is_numeric = False if type(vals[0]) is str else True

    def val_format(val):
        return str(val) if is_numeric else '"' + str(val) + '"'

    if_pat = 'IF({cond}, {th}, {el})'
    clause_ls = []
    for i, thr in enumerate(bins):
        then_clause = val_format(vals[i])
        else_clause = val_format(vals[i+1]) if i == len(bins) - 1 else '\n{el}'
        clause_ls.append(if_pat.format(cond=o2 + " < " + str(thr), th=then_clause, el=else_clause))

    clause_ls = clause_ls[::-1]
    o_subquery = clause_ls[0]
    for clause in clause_ls[1:]:
        o_subquery = clause.format(el=o_subquery)
    o1 = "?" + fun_name + "_" + o1[1:] if fun_name else o1    # agg: agg_design_attribute for tracking functions
    o_subquery = ('BIND(\n{if_clause} AS ' + o1 + '\n).').format(if_clause=o_subquery)

    return [' '.join([s, p, o2])+'.', o_subquery], o1


def combine_if_clause(o2, comb_names: list, vals: list, priority: bool):
    if_patn = 'IF(regex({o2}, "({or_val})"), \n\t{then_clause}, \n\t{else_clause})'
    len_ = len(comb_names)
    qry2 = ''
    assert len(comb_names) == len(vals) and len(vals) > 0

    def fill_if(pat: str):
        if priority:
            return pat.format(o2=o2,
                              or_val='|'.join([";" + str(v) + ";" for v in has_vals]),
                              then_clause='"{}"'.format(new_val),
                              else_clause=if_patn)
        else:
            return pat.format(o2=o2,
                              or_val='|'.join([";" + str(v) + ";" for v in has_vals]),
                              then_clause=if_patn.format(o2=o2,
                                                         or_val='|'.join([";" + str(v) + ";" for v in not_has_vals]),
                                                         then_clause='"Other"',
                                                         else_clause='"{}"'.format(new_val)),
                              else_clause=if_patn)

    for i, new_val in enumerate(comb_names):
        has_vals = vals[i]
        not_has_vals = None if i + 1 == len_ else reduce(operator.concat, vals[i+1:])
        if i == 0:
            if len_ == 1:
                qry2 = if_patn.format(o2=o2,
                                   or_val='|'.join([";" + str(v) + ";" for v in has_vals]),
                                   then_clause='"{}"'.format(new_val),
                                   else_clause='"Other"')
            else:
                qry2 = fill_if(if_patn)
        elif i + 1 < len_:
            qry2 = fill_if(qry2)
        else:
            qry2 = qry2.format(o2=o2,
                               or_val='|'.join([";" + str(v) + ";" for v in has_vals]),
                               then_clause='"{}"'.format(new_val),
                               else_clause='"Other"')
    return qry2


def combine_clause(s: str, p: str, o1: str, o2: str, o3: str, comb_names: list, vals: list, priority=False, fun_name=''):
    '''
    :param s:
    :param p:
    :param o1:
    :param o2:
    :param o3:
    :param comb_names:
    :param vals:
    :param priority:
    :param fun_name: fun_name != '' when agg use this function
    :return:
    '''
    subquery_patn = 'SELECT {s1} (CONCAT(";", str(GROUP_CONCAT({o3}; SEPARATOR=";")), ";") AS {o2}) WHERE {where} GROUP BY {s2}'
    qry1 = '{' + subquery_patn.format(s1=s,
                                      o3=o3,
                                      o2=o2,
                                      where='{\n' + ' '.join([s, p, o3]) + '.\n}',
                                      s2=s) + '}'       # s1, o2

    qry2 = combine_if_clause(o2, comb_names, vals, priority)

    o1 = "?" + fun_name + "_" + o1[1:] if fun_name else o1  # agg: agg_design_attribute for tracking functions
    qry2 = 'BIND({if_cond} AS {o1}).'.format(if_cond=qry2, o1=o1)
    return [qry1, qry2], o1


def combine_clause4agg(o1: str, comb_names: list, vals: list, priority=False, fun_name=''):
    '''
    :param o1:
    :param comb_names:
    :param vals:
    :param priority:
    :param fun_name: fun_name != '' when agg use this function
    :return:
    '''
    concat_o1 = 'CONCAT(";", str(GROUP_CONCAT({o1}; SEPARATOR = ";")), ";")'.format(o1=o1)
    qry = combine_if_clause(concat_o1, comb_names, vals, priority)
    o1 = "?" + fun_name + "_" + o1[1:] if fun_name else o1  # naming agg_design_attribute for tracking functions
    return '({qry} AS {var})'.format(qry=qry, var=o1)


def fill_query(query_pattern, prefix, head, body, tail=None):
    if not tail:
        query_pattern = query_pattern.replace(" GROUP BY {tail}", "{tail}")
        tail = []
    return query_pattern.format(prefix='' if prefix is None else '\n'.join(
        ["PREFIX " + bref_url + ": <" + full_url + ">" for full_url, bref_url in prefix.items()]) + '\n',
                              head=' '.join(head),
                              body='{\n' + '\n'.join(body) + '\n}',
                              tail=' '.join(tail))


class QueryEngine:

    def __init__(self, endpoint_url: str):
        '''
        :param endpoint_url: the url of your sparql endpoint url
        '''

        self.sparql_url = endpoint_url
        self.sparql_engine = SPARQLWrapper(self.sparql_url)
        self.sparql_engine.setReturnFormat(JSON)

    def query(self, query: str):
        '''
        :param query: sparql query
        :return: JSON
        '''
        self.sparql_engine.setQuery(query)
        ret = self.sparql_engine.queryAndConvert()
        return ret


class ConceptMapping:
    def __init__(self, names: list, uris: list, types: list):
        '''
        :param uris: e.g. [http://www.w3.org/TR/2003/PR-owl-guide-20031209/wine#Medoc, ...]
        :param names: e.g. [Medoc, ...]
        :param types: e.g. [e, ..., ], where e can be 0:class, 1:property, 2:attribute
        when name is a constant attribute, uris is data type
        '''
        assert len(names) == len(uris)

        assert len(names) == len(set(names))

        self.classes, self.properties, self.attributes = set(), set(), set()
        for c, t in zip(names, types):
            if t == 0:
                self.classes.add(c)
            elif t == 1:
                self.properties.add(c)
            elif t == 2:
                self.attributes.add(c)
            else:
                raise Exception("Wrong concept type for", c, "not in [0,1,2]")

        self.uri2name_dict = {k: v for k, v in zip(uris, names) if type(k) is str}
        self.name2uri_dict = {v: k for v, k in zip(names, uris) if type(k) is str}
        self.id2name_dict = {i: name for i, name in enumerate(names)}
        self.id2uri_dict = {v: self.name2uri_dict[k] for v, k in self.id2name_dict.items() if
                            k in self.name2uri_dict.keys()}
        self.name2id_dict = {name: i for i, name in enumerate(names)}
        self.id2type_dict = [None if type(e) is str else e for e in uris]  # uri is None
        self.name2type_dict = {k: self.id2type_dict[v] for k, v in self.name2id_dict.items()}  # uri is None

    def name2uri(self, name: str):
        if name[0] == '?':
            name = name[1:]
        if name not in self.name2uri_dict.keys():
            return None
        return self.name2uri_dict[name]     # if name in self.name2uri_dict.keys() else None

    def uri2name(self, uri: str):
        return self.uri2name_dict[uri]  # if uri in self.uri2name_dict.keys() else None

    def id2name(self, _id: int):
        return self.id2name_dict[_id]   # if _id in self.id2name_dict.keys() else None

    def id2uri(self, _id: int):
        return self.id2uri_dict[_id]    # if _id in self.id2uri_dict.keys() else None

    def name2id(self, name: str):
        if name[0] == '?':
            name = name[1:]
        return self.name2id_dict[name]  # if name in self.name2id_dict.keys() else -1

    def id2type(self, _id: int):
        return self.id2type_dict[_id]   # if _id in self.id2name_dict.keys() else None

    def name2type(self, name: str):
        if name[0] == '?':
            name = name[1:]
        return self.name2type_dict[name]    # if name in self.name2type_dict.keys() else None


class QueryBuilder:

    def __init__(self,
                 query_engine: QueryEngine,
                 concept_mapping: ConceptMapping,
                 graph_structure: List[Tuple[Any, Any, Any]],
                 prefix: dict = None):
        '''
        :param concept_mapping:
        :param graph_structure: list of triple [(subject, predicate, object)], each element in triple can be int or name
        :param prefix: dict={full: brief} prefix used in sparql query
        # TODO 1. when relation has attributes
        # TODO 2. allow using optional over attributes

        '''

        # inputs
        self.query_engine = query_engine
        self.map = concept_mapping
        self.graph_structure = [(self.map.id2name(s), self.map.id2name(p), self.map.id2name(o)) for s, p, o in graph_structure] if type(
            graph_structure[0][0]) is int else graph_structure

        self.funs = {'avg': 'AVG({})',
                     'sum': 'SUM({})',
                     'max': 'MAX({})',
                     'min': 'MIN({})',
                     'count': 'COUNT({})',
                     'count_distinct': 'COUNT(DISTINCT {})'}
        # 'concat': 'GROUP_CONCAT({}; SEPARATOR = ";")',
        # 'concat_distinct': 'GROUP_CONCAT(DISTINCT {}; SEPARATOR = ";")'}

        self.num_funs = set(['avg', 'sum', 'max', 'min', 'count', 'count_distinct'])
        self.cat_funs = set(['count', 'count_distinct'])
        self.sup_var_type = [int, float, str]

        # query
        self.query_pattern = "{prefix}SELECT DISTINCT {head} WHERE {body} GROUP BY {tail}"
        self.prefix = prefix

        # concept group
        # self.classes, self.properties, self.attributes, self.att_multi_val = set(), set(), set(), dict()
        self.att_multi_val = dict()
        # self.concept_group()

        self.unit_identifiers = set()  # identifiers format unit
        self.identifier_as_att = set()

        for att in self.map.attributes:
            self.att_multi_val[att] = False
        # attribute is multi (True) or single (False) type
        # self.multi_single_attr()
        self.unique_values_attr()

    def class_uri(self, c):
        if type(c) is int:
            c = self.map.id2name(c)
        assert c in self.map.classes
        return self.map.name2uri(c)

    def brief_format(self, c1):
        if not c1:
            return c1
        if '/' not in c1:
            return c1
        c = c1
        if '<' == c[0]:
            c = c[1:]
        if '>' == c[-1]:
            c = c[:-1]
        pos = c.rfind('/') + 1
        pre, short_c = c[:pos], c[pos:]
        if pre in self.prefix.keys():
            return self.prefix[pre] + ":" + short_c
        return c1

    def multi_single_attr(self):
        select_head = set()
        where_body = []
        class_type_body = set()
        for _s, _p, _o in self.graph_structure:
            if _o not in self.map.attributes:
                continue

            s = '?' + _s
            o = '?' + _o
            p = self.brief_format(self.map.name2uri(_p))
            o1 = o + '1'
            o2 = o + '2'
            if self.map.name2uri(_s):
                class_type_body.add((s + " a " + self.brief_format(self.class_uri(_s)) + "."))

            where_body.append('{SELECT (COUNT('+o2+') AS '+o1+') WHERE {'+' '.join([s, p, o2])+'.' +'} GROUP BY '+s+'}')
            select_head.add('(MAX({}) AS {})'.format(o1, o))

        qry = fill_query(self.query_pattern,
                         prefix=self.prefix,
                         head=select_head,
                         body=[*class_type_body, *where_body])
        print("QueryBuilder.multi_single_attr():\n"+qry+"\n")
        df = json2df(self.query_engine.query(qry))

        if df.empty:

            return

        for col in df.columns:
            self.att_multi_val[col] = df[col][0] > 1

        print("Multi-Value Attributes:", [att for att, multi in self.att_multi_val.items() if multi])
        print("Single-Value Attributes:", [att for att, multi in self.att_multi_val.items() if not multi])
        print('\n')

    def unique_values_attr(self):

        for _s, _p, _o in self.graph_structure:
            if _o not in self.map.attributes:
                continue

            s = '?' + _s
            o = '?' + _o
            p = self.brief_format(self.map.name2uri(_p))
            where_body = []
            if self.map.name2uri(_s):
                where_body.append(s + " a " + self.brief_format(self.class_uri(_s)) + ".")

            where_body += [' '.join([s, p, o])+"."]

            qry = fill_query(self.query_pattern,
                             prefix=self.prefix,
                             head=[o],
                             body=where_body)

            df = json2df(self.query_engine.query(qry))
            print(o, " unique values: ", end='')
            print(df[_o].values.tolist())

        print('\n')

    def exception_check(self, _var, _fun):
        if _fun in self.funs.keys():
            if self.map.name2type(_var) not in self.sup_var_type:
                raise Exception("The data type of ", _var, "is not supported in aggregation functions.")
            if self.map.name2type(_var) is str:
                # print(_fun, _var)
                if _fun not in self.cat_funs:
                    raise Exception("Error for attribute ", _var, "categorical only allows following functions",
                                    self.cat_funs)
            else:  # [int, float]
                if _fun not in self.num_funs:
                    raise Exception("Error for attribute ", _var, "numerical in only allows following functions",
                                    self.num_funs)
        elif _fun == 'disc':
            if self.map.name2type(_var) not in [int, float]:
                raise Exception("Error for attribute ", _var, ": Discretization is only for int or float")
        elif _fun == 'repl':
            if self.map.name2type(_var) not in [int, str]:
                raise Exception("Error for attribute ", _var, ": Replacement is only for str")
        elif _fun == 'comb':
            if self.map.name2type(_var) not in [int, str]:
                raise Exception("Error for attribute ", _var, ": Combination is only for str")
        else:
            raise Exception("Unknown design function:", _fun, "of Varialble", _var)

    def sparql_query(self, perspective: list, agg_strategy: dict, attr_design=None, invalid_entities: dict = None):
        '''
        # :param track_fun: whether keep the function name in variables? True or False
        :param perspective: list of str (includes classes, and relations) or int (position of classes or relation in concept_mapping)
        :param invalid_entities: {'identity_name':[], ..., }
        :param agg_strategy: dict where key is str or int (represent attribute), and value is list of function
        :param attr_design: agg, replace; disc; comb; repl.  aggregation√, discretize√, combination(result in binary value).
        parameters example for agg_strategy and attr_design
            'numerical agg': threshold
            'disc':  # {'bins': [60, 80], 'vals': [0, 1, 2]})
            'repl':  # {'old_vals': ['ALK', 'EGFR'], 'new_vals': ['alk', 'egfr']}
            'comb':  # {'comb_name': 'ALKorEGFR', 'comb_vals': ['ALK', 'EGFR'], 'label_empty': True}
        :return: sparql query
        '''

        agg_strategy = {self.map.id2name(k): v for k, v in agg_strategy.items()} if type(
            list(agg_strategy.keys())[0]) is int else agg_strategy
        attr_design = None if attr_design is None else ({self.map.id2name(k): v for k, v in attr_design.items()} if type(
            list(attr_design.keys())[0]) is int else attr_design)

        if invalid_entities:
            invalid_entities = {self.map.id2name(k) if type(k) is int else k: [self.brief_format(v) for v in vals] for k, vals in invalid_entities.items()}

        perspective = [self.map.id2name(e) for e in perspective] if type(perspective[0]) is int else perspective
        self.unit_identifiers = set([e for e in perspective if e in self.map.classes])  # group by unit identifier
        select_head = set()
        where_body = []
        class_type_body = set()
        group_by_tail = set()

        def add_identity(var):
            assert '?' == var[:1]

            if self.map.name2uri(var[1:]):      # add class type
                if var[1:] in self.unit_identifiers:
                    class_type_body.add(var + " a " + self.brief_format(self.class_uri(var[1:]))+".")
                else:
                    class_type_body.add(var + '1' + " a " + self.brief_format(self.class_uri(var[1:])) + ".")

            if var[1:] in self.unit_identifiers:     # [1:] to not consider "?", group by identifier
                select_head.add(var)
                group_by_tail.add(var)
                return var
            else:                               # concat group identifier
                # for other class identifiers
                # if identity used as feature
                if var[1:] in agg_strategy.keys():
                    self.identifier_as_att.add(var[1:])
                    paras = agg_strategy[var[1:]]['comb']
                    agg_var = combine_clause4agg(var, paras['comb_names'], paras['comb_vals'], paras['prior'])
                    select_head.add(agg_var)
                # else
                else:
                    select_head.add('(' + 'GROUP_CONCAT(DISTINCT {}; SEPARATOR = ";")'.format(var + '1') + ' AS ' + var + ')')
                return var + '1'

        def rm_invalid_clause(s_or_o: str, entity_list: list):
            return "FILTER({s} NOT IN ({ls}))".format(s=s_or_o,
                                               ls=','.join(entity_list if "<" in entity_list[0] else [e if ':' in e else "<{}>".format(e) for e in entity_list]))

        for _s, _p, _o in self.graph_structure:
            s = '?' + _s
            o = '?' + _o
            o1 = o + "1"
            self.map.name2type_dict[o1[1:]] = self.map.name2type(_o)    # in case attr_design is None
            p = self.brief_format(self.map.name2uri(_p))

            if _o in self.map.classes:   # (class, relation, class)
                s = add_identity(s)
                o = add_identity(o)
                where_body.append(' '.join([s, p, o])+'.')

                if invalid_entities:  # remove invalid entities
                    if _s in invalid_entities.keys():
                        where_body.append(rm_invalid_clause(s, invalid_entities[_s]))
                        del invalid_entities[_s]
                    if _o in invalid_entities.keys():
                        where_body.append(rm_invalid_clause(o, invalid_entities[_o]))
                        del invalid_entities[_o]
            else:                   # (class, property, attribute)
                s = add_identity(s)

                design_flag = (True if _o in attr_design.keys() else False) if attr_design else False
                agg_flag = True if _o in agg_strategy.keys() else False

                if design_flag:             # TODO for each ?s:  -> o1
                    o2 = o + '2'            # in where body
                    assert len(attr_design[_o]) == 1    # only one fuction is allowed for design
                    design_option, paras = '', ''
                    for k, v in attr_design[_o].items():
                        design_option, paras = k, v     # [design_option, (paras, ...)]

                    o1 = ('?'+design_option+'_'+o[1:])  # if track_fun else o+"1"                # o1 = o + '1'

                    self.exception_check(_o, design_option)
                    if design_option in self.funs.keys():   # agg: paras = None
                        clause = 'SELECT {s1} ({agg_o2} AS {o1}) WHERE {where} GROUP BY {s2}'
                        clause = '{'+clause.format(s1=s,
                                                   agg_o2=self.funs[design_option].format(o2),
                                                   o1=o1,
                                                   where='{\n'+' '.join([s, p, o2])+'.\n}',
                                                   s2=s)+'}'
                        clauses = [clause]
                        self.map.name2type_dict[o1[1:]] = int if 'count' in design_option else float
                    elif design_option == 'disc':
                        clauses, _ = discretize_clause(s, p, o1, o2, paras['bins'], paras['vals'])
                        self.map.name2type_dict[o1[1:]] = type(paras['vals'][0])    # update o1 data type
                    elif design_option == 'repl':
                        clauses, _ = replace_clause(s, p, o1, o2, paras['old_vals'], paras['new_vals'])
                        self.map.name2type_dict[o1[1:]] = type(paras['new_vals'][0])    # update o1 data type
                    elif design_option == 'comb':
                        clauses, _ = combine_clause(s, p, o1, o2, o+"3", paras['comb_names'], paras['comb_vals'], paras['prior'])
                        self.map.name2type_dict[o1[1:]] = type(paras['comb_names'][0])  # update o1 data type
                    where_body = [*where_body, *clauses]
                else:
                    where_body.append(' '.join([s, p, o1])+'.')

                if agg_flag: # TODO for each unit identifier: o1 -> o
                    if _s in self.unit_identifiers and not self.att_multi_val[_o]:  # single-value attribute of unit id
                        print(_o, "is a single-value attribute of a unit identifier, no need aggregation, pass!")
                        select_head.add(o)
                        where_body.append('BIND({o1} AS {o}).'.format(o1=o1, o=o))
                        group_by_tail.add(o)
                        continue

                    # o1 = o + '1'            # in where body
                    for o_fun in agg_strategy[_o]:
                        self.exception_check(o1[1:], o_fun)     # here should check o1

                        paras = agg_strategy[_o][o_fun]
                        if o_fun in self.funs.keys():   # todo threshold
                            select_head.add('({agg_o1} AS {o})'.format(agg_o1=self.funs[o_fun].format(o1), o=('?'+o_fun+"_"+o[1:])))   # if track_fun else o))   # naming agg_design_attribute
                        elif o_fun == 'comb':   # comb is different
                            agg_o = combine_clause4agg(o1, paras['comb_names'], paras['comb_vals'], paras['prior'], o_fun)
                            select_head.add(agg_o)
                        else:
                            if o_fun == 'disc':
                                if self.att_multi_val[_o]:  # unique exception for agg
                                    raise Exception("Discretization is not allowed for", _o, " please applly a multi -> single function")
                                clauses, agg_o = discretize_clause(s, p, o, o1, paras['bins'], paras['vals'], o_fun)
                            elif o_fun == 'repl':
                                if self.att_multi_val[_o]:  # unique exception for agg
                                    raise Exception("Replacement is not allowed for", _o, " please applly a multi -> single function")
                                clauses, agg_o = replace_clause(s, p, o, o1, paras['old_vals'], paras['new_vals'], o_fun)
                            where_body = [*where_body, *clauses]
                            select_head.add(agg_o)
                            group_by_tail.add(agg_o)

                    # if not design_flag:     # no design, no o1
                    #     where_body.append(' '.join([s, p, o1])+'.')
                else:                       # put o into tail group by tail, and select head
                    if _s not in self.unit_identifiers:
                        raise Exception(_s, "is not an unit identifier,", _o, " must be aggregated.")
                    if self.att_multi_val[_o]:
                        raise Exception(_o, "is a multi-value attribute of ", _s, ", must be aggregated.")
                    # only single-value attribute of any unit identifier allowed to pass through
                    # if design_flag:
                    select_head.add(o)   # no element for body
                    where_body.append('BIND({o1} AS {o}).'.format(o1=o1, o=o))  # o1 -> o
                    # else:
                    #     where_body.append(' '.join([s, p, o])+'.')  # no o1
                    #     select_head.add(o)
                    group_by_tail.add(o)
        where_body = [*class_type_body, *where_body]
        return fill_query(self.query_pattern, self.prefix, select_head, where_body, group_by_tail)

    def constraint_query(self, target_class, constraints: dict=None):
        '''
        :param target_class: 
        :param constraints: 
        Conjunction conditions:
        "min_card": dict={"path": [[], ...] , "card":[int, ...]},
        "max_card": dict={"path": [[], ...] , "card":[int, ...]},
        "min_val": dict={"path": [[], ...] , "val":[int, ...]},
        "max_val": dict={"path": [[], ...] , "val":[int, ...]},
        "has_and_vals": dict={"path": [[], ...] , "vals":[[], ...]},
        "has_or_vals": dict={"path": [[], ...] , "vals":[[], ...]},
        "class": dict={"is": [], "not": []}, 
        
        :return: 
        '''
        # TODO checking valid path end at class or property
        if constraints is None:
            return None

        def ls_dig2str(path):
            assert len(path) > 2 and len(path) % 2 == 1
            if type(path[0]) is int:
                path = [self.map.id2name(e) for e in path]
            assert path[0] in self.map.classes  # the first one should be class
            return path

        def path_triples(path):
            return [' '.join(['?' + _s1, self.brief_format(self.map.name2uri(_p)), '?' + _o1]) + '.' for _s1, _p, _o1 in
             zip(path[:-2][::2], path[1:-1][::2], path[2:][::2])]

        def card_constraint(s, path:list, card:int, max_flag:bool):      # max_card, min_card
            path = ls_dig2str(path)
            if s[0] != '?':
                s = '?'+s
            o = '?'+path[-1]
            o1 = '?'+path[-1]+'1'
            head = [s]
            body = [s+" a "+self.map.name2uri(s[1:])+"."]   # without ?

            sub_qry = "SELECT {s1} (COUNT({o}) AS {o1}) WHERE {where} GROUP BY {s2}"
            body.append("{"+sub_qry.format(s1=s, o=o, o1=o1, where='{\n'+'\n'.join(path_triples(path))+"}\n", s2=s)+"}")

            filter_clause = 'FILTER({o1} {cond} {val}).'.format(o1=o1, cond='>' if max_flag else "<", val=str(card))
            body.append(filter_clause)
            return head, body

        def val_range_constraint(s, path:list, val, max_flag:bool):     # max_val, min_val
            assert type(val) is not str
            path = ls_dig2str(path)
            if s[0] != '?':
                s = '?' + s
            o = '?' + path[-1]
            o1 = '?' + path[-1] + '1'
            head = [s]
            body = [s + " a " + self.map.name2uri(s[1:]) + "."]

            sub_qry = "SELECT {s1} ({fun}({o}) AS {o1}) WHERE {where} GROUP BY {s2}"
            body.append("{"+sub_qry.format(s1=s, fun="MAX" if max_flag else "MIN", o=o, o1=o1, where='{\n' + '\n'.join(path_triples(path)) + "}\n", s2=s)+"}")

            filter_clause = 'FILTER({o1} {cond} {val}).'.format(o1=o1, cond='>' if max_flag else "<", val=str(val))
            body.append(filter_clause)
            return head, body

        def has_vals_constraint(s, path:list, vals, and_flag:bool):     # has_val
            vals = [str(val) if type(vals) is not str else '"{}"'.format(str(val)) for val in vals]
            path = ls_dig2str(path)
            # if exist then should be in the limited values
            if s[0] != '?':
                s = '?' + s
            o = '?' + path[-1]
            head = [s]
            body = [s + " a " + self.map.name2uri(s[1:]) + "."]     # without ?
            # body = [*body, *path_triples(path)]
            path_trips = path_triples(path)
            if and_flag:    # and
                path_trips, last_triple = path_trips[:-1], path_trips[-1]
                pos1 = last_triple.rfind('?')
                pos2 = last_triple[:pos1].rfind('?')

                rest_trips = last_triple[pos2-1:]
                patn = last_triple[pos2:pos1]    # ?s ?p ####
                trips = [patn+str(val)+"." for val in vals]
                path_trips = [*path_trips, *trips]
                body.append("FILTER NOT EXISTS {"+"\n".join(path_trips)+"\n}")

            else:   # or
                clauses = '\n'.join(['\n'.join(path_trips), 'FILTER ({o} IN ({vals}))'.format(o=o, vals=','.join(vals))])
                body.append("FILTER NOT EXISTS {"+clauses+"\n}")

            # conds = ["{o} != {val}".format(o=o, val=val) for val in vals]
            # filter_clause = "FILTER NOT EXISTS {("+ (' && ' if and_flag else ' || ').join(conds) +")}"
            # body.append(filter_clause)
            return head, body

        def class_constraint(s, is_classes, not_classes):           # is classes, not classes
            assert len(is_classes) > 0
            if s[0] != '?':
                s = '?' + s
            s_uri = self.brief_format(self.class_uri(s[1:]))    # s must have class concept
            head = [s]
            body = [s + " a " + s_uri + "."]

            is_classes = [self.brief_format(self.class_uri(c)) for c in is_classes]
            not_classes = [self.brief_format(self.class_uri(c)) for c in not_classes]

            if s_uri in is_classes:
                is_classes.remove(s_uri)

            cond = " || ".join(
                ["NOT EXISTS{" + ' '.join([s, 'a', c]) + "}" for c in is_classes])
            filter_clause = 'FILTER({}).'.format(cond)
            if is_classes:
                body.append(filter_clause)

            cond = " || ".join(["EXISTS{"+' '.join([s, 'a', c])+"}" for c in not_classes])
            filter_clause = 'FILTER({}).'.format(cond)
            if not_classes:
                body.append(filter_clause)
            return head, body

        _s = self.map.id2name(target_class) if type(target_class) is int else target_class
        s = '?' + _s
        head = []
        body = []
        for constraint_type, paras in constraints.items():
            if constraint_type in ['max_card', 'min_card']:
                assert type(paras['path']) is list and type(paras['card']) is list
                assert len(paras['path']) == len(paras['card'])
                head1, body1 = [], []
                for path, card in zip(paras['path'], paras['card']):
                    head_, body_ = card_constraint(s, path, card, max_flag=True if constraint_type == 'max_card' else False)
                    head1 = [*head1, *head_]
                    body1 = [*body1, *body_]
            elif constraint_type in ['min_val', 'max_val']:
                assert type(paras['path']) is list and type(paras['val']) is list
                assert len(paras['path']) == len(paras['val'])
                head1, body1 = [], []
                for path, val in zip(paras['path'], paras['val']):
                    head_, body_ = val_range_constraint(s, path, val, max_flag=True if constraint_type == 'max_val' else False)
                    head1 = [*head1, *head_]
                    body1 = [*body1, *body_]
            elif constraint_type in ['has_and_vals', 'has_or_vals']:
                assert type(paras['path']) is list and type(paras['vals']) is list
                assert len(paras['path']) == len(paras['vals'])
                head1, body1 = [], []
                for path, vals in zip(paras['path'], paras['vals']):
                    head_, body_ = has_vals_constraint(s, path, vals, and_flag=True if constraint_type == 'has_and_vals' else False)
                    head1 = [*head1, *head_]
                    body1 = [*body1, *body_]
            elif constraint_type == 'class':
                head1, body1 = class_constraint(s, paras['is'], paras['not'])
            head = [*head, *head1]
            body = [*body, *body1]
        return fill_query(self.query_pattern, self.prefix, head, body)

    @property
    def unit_ids(self):
        if self.unit_identifiers is None:
            raise Exception("Need perspective and execute sparql_query()")
        return self.unit_identifiers

    @property
    def other_ids(self):
        if self.unit_identifiers is None:
            raise Exception("Need perspective and execute sparql_query()")
        return self.map.classes.difference(self.unit_identifiers)


def test_data_layer():
    sparql_engine = QueryEngine(endpoint_url='http://localhost:7200/repositories/university')
    prefix = '<http://www.example.com/{}>'
    id2concept = {0: 'prof', 1: 'univ', 2: 'stu', 3: 'hasRank', 4: 'workIn', 5: 'supervise', 6: 'hasScore',
                  7: 'rank', 8: 'score', 9: 'hasMutation', 10: 'mutation'}
    map = ConceptMapping(['prof', 'univ', 'stu', 'hasRank', 'workIn', 'supervise', 'hasScore', 'rank', 'score', 'hasMutation', 'mutation'],
                         [prefix.format('Professor'), prefix.format('University'), prefix.format('Student'), prefix.format('hasRank'), prefix.format('workIn'), prefix.format('supervise'), prefix.format('hasScore'), int, float, prefix.format('hasMutation'), str])
    structure = [(0, 4, 1), (1, 3, 7), (0, 5, 2), (2, 6, 8), (2, 9, 10)]

    querybuilder = QueryBuilder(query_engine=sparql_engine, concept_mapping=map, graph_structure=structure)

    # TODO test Design
    # attr_design = {}
    # attr_design = {8: ('disc', {'bins': [60, 80], 'vals': [0, 1, 2]})}
    # attr_design = {10: ('repl', {'old_vals': ['ALK', 'EGFR'], 'new_vals': ['alk', 'egfr']})}
    # attr_design = {10: ('comb', {'comb_name': 'ALKorEGFR', 'comb_vals': ['ALK', 'EGFR'], 'label_empty': True})}
    attr_design = {8: ('avg', None),
                   10: ('comb', {'comb_name': 'ALKorEGFR', 'comb_vals': ['ALK', 'EGFR'], 'label_empty': True})}

    # TODO test collect by agg
    agg_strategy = {8: {'max': None},
                    10: {'comb': {'comb_name': 'ALKorEGFR', 'comb_vals': ['ALKorEGFR'], 'label_empty': True}}}
    # query = querybuilder.sparql_query(attr_design=None, perspective=[0, 1], agg_strategy=agg_strategy)

    # TODO test constraint query
    # constraints = {"min_card": {"path": [[2, 6, 8]], "card": [2]}}
    # constraints = {"max_card": {"path": [[2, 6, 8]], "card": [2]}}
    # constraints = {"max_val": {"path": [[2, 6, 8]], "val": [80]}}
    # constraints = {"min_val": {"path": [[2, 6, 8]], "val": [80]}}
    # constraints = {"has_and_vals": {"path": [[2, 6, 8]], "vals": [[76, 82]]}}
    # constraints = {"has_or_vals": {"path": [[2, 6, 8]], "vals": [[76, 62]]}}
    constraints = {"class": {"is": [2], "not": [0]}}

    query = querybuilder.constraint_query(target_class=2, constraints=constraints)
    print(query)
    df = json2df(sparql_engine.query(query))
    df.to_csv('tmp.csv')


def test_mapping():
    mapping = ConceptMapping(['man', 'hasAdog', 'dog', 'age', 'name'],
                             ['http://man', 'http://hasAdog', 'dog', int, str])
    print(mapping.id2name(1))
    print(mapping.id2type(3))
    print(mapping.name2uri('man'))


def test_sparql():
    sparql = SPARQLWrapper("http://localhost:7200/repositories/wine")
    sparql.setReturnFormat(JSON)

    # gets the first 3 geological ages
    # from a Geological Timescale database,
    # via a SPARQL endpoint
    sparql.setQuery("""
        PREFIX a: <http://www.example.com/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        select ?p ?rank (max(?avgscore) as ?maxavgscore) where{
        ?p a:supervise ?s.
        ?p a:workIn ?unv.
        ?unv a:hasRank ?rank.
        {select ?s (avg(xsd:float(?score)) as ?avgscore) where { 
        ?s a:hasScore ?score.} 
        group by ?s}
        } group by ?p ?rank
        """)
    try:
        ret = sparql.queryAndConvert()
        print(json2df(ret))
        # for r in ret["results"]["bindings"]:
        #     print(r)
    except Exception as e:
        print(e)
