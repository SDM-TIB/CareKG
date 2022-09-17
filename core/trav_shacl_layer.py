import os
from argparse import Namespace
from TravSHACL.core.ShapeSchema import ShapeSchema
from TravSHACL.core.GraphTraversal import GraphTraversal

constraints_dir = '../config/shacl/'    # each file is a json of constriant
output_dir = "../output/"
endpoint = 'http://localhost:7200/repositories/university'

trav_args = {
        'd': os.path.realpath(constraints_dir),         # constraint schema dir
        'endpoint': endpoint,                           # endpoint
        'graphTraversal': "DFS",                        # "DFS", "BFS"
        'heuristics': "TARGET IN BIG",  # params.get('heuristic') or DEFAULT_PARAMS['heuristic'],
        'm': 256,       # json_config.get('maxSplit') or def_config['maxSplit'],
        'orderby': True,        # json_config.get('ORDERBYinQueries') or def_config['ORDERBYinQueries'],
        'outputDir': output_dir,        # json_config.get('outputDirectory') or def_config['outputDirectory'],
        's2s': False,   # json_config.get('SHACL2SPARQLorder') or def_config['SHACL2SPARQLorder'],
        'selective': True,   # json_config.get('useSelectiveQueries') or def_config['useSelectiveQueries'],
        'outputs': False,       # json_config.get('outputs') or def_config['outputs'],
        'json': True
    }


def parse_heuristics(input_):
    """
    Parses the heuristics from the arguments passed to Trav-SHACL.
    :param input_: the heuristics argument
    :return: Python dictionary with the heuristics to be used for the evaluation
    """
    heuristics = {}
    if 'TARGET' in input_:
        heuristics['target'] = True
    else:
        heuristics['target'] = False

    if 'IN' in input_:
        heuristics['degree'] = 'in'
    elif 'OUT' in input_:
        heuristics['degree'] = 'out'
    else:
        heuristics['degree'] = False

    if 'SMALL' in input_:
        heuristics['properties'] = 'small'
    elif 'BIG' in input_:
        heuristics['properties'] = 'big'
    else:
        heuristics['properties'] = False

    return heuristics


# def parse_shacl_report(report):


def get_shacl_result(args):
    # print(args)
    shape_schema = ShapeSchema(schema_dir=args.d,
    schema_format='JSON' if args.json else 'SHACL',
    endpoint_url=args.endpoint,
    graph_traversal=GraphTraversal.BFS if args.graphTraversal == 'BFS' else GraphTraversal.DFS,
    heuristics=parse_heuristics(args.heuristics),
    use_selective_queries=args.selective,
    max_split_size=args.m,
    output_dir=args.outputDir,
    order_by_in_queries=args.orderby,
    save_outputs=args.outputs)
    report = shape_schema.validate()
    return report


print(get_shacl_result(Namespace(**trav_args)))

