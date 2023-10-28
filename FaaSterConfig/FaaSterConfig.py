#!/bin/sh
"""":
if type python3 > /dev/null 2>&1
then
    exec python3 "$0" "$@"
elif type python > /dev/null 2>&1
then
    exec python "$0" "$@"
else
    echo >&2 "Python not installed"
fi
exit 1
""" #"

import os
import numpy as np
import random
import yaml
from concurrent.futures import ThreadPoolExecutor  # or ProcessPoolExecutor if you want to use multiple processes
from doepy import build
import subprocess
import re
from tabulate import tabulate


global_config_space = {
    'CPU':[.5,4,16],
    'Mem':[248, 1024, 1024*16],
    'NodeType':["GPU1", "GPU2", "GPU3"]
}

# Helpers
def executeThreaded(func, args, max_workers=4):
    """Executes a function over a list of arguments in a threaded way 

    Args:
        func (Callable): The function that will be called on each args elem to produce a new value
        args: An array of arguments to pass to func
        max_workers (int): Number of threads to run at once

    Returns:
        df (pandas.DataFrame): The original dataframe with the new column
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, args))

    return results

def readStack(path='../openFaas/stack.yml'):
    """
    Read and parse a YAML file containing an OpenFaaS stack definition.

    This function opens and reads a YAML file located at the specified 'path' and parses its contents
    using the PyYAML library. The resulting data is typically an OpenFaaS stack definition.

    Parameters:
    path (str, optional): The path to the YAML file to be read (default is '../openFaas/stack.yml').

    Returns:
    dict: A Python dictionary representing the parsed contents of the YAML file.
    """
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
        
    return data
    
def writeStack(data, path='../openFaas/stack.yml'):
    """
    Write a Python dictionary as a YAML file representing an OpenFaaS stack definition.

    This function takes a Python dictionary 'data' and writes it to a YAML file at the specified 'path'
    using the PyYAML library. The dictionary is typically used to represent an OpenFaaS stack definition.

    Parameters:
    data (dict): The Python dictionary representing the OpenFaaS stack definition to be written.
    path (str, optional): The path to the output YAML file (default is '../openFaas/stack.yml').
    """
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def getConfigResultsFaked(row, sigma=.1):
    """
    Generate faked configuration results based on input row data.

    This function generates synthetic configuration results by using random values based on the input data
    in the 'row' argument. It can be useful for testing and simulation purposes.

    Parameters:
    row (pd.Series): A Pandas Series containing the input data used to generate the synthetic results.
    sigma (float, optional): The standard deviation for random value generation (default is 0.1).

    Returns:
    float: A synthetic configuration result based on the input data.

    Example:
    >>> import pandas as pd
    >>> data = {'i': 42, 'CPU': 4, 'Mem': 16, 'NodeType': 'A'}
    >>> row = pd.Series(data)
    >>> result = getConfigResultsFaked(row, sigma=0.2)
    >>> print(result)
    # Output is a synthetic configuration result based on the input data and randomness.
    """
    random.seed(row.i)
    b = random.normalvariate(.5, sigma)
    return b + random.normalvariate(1/(row.CPU * 3 + row.Mem*1.5 * row.NodeType), sigma)

def genConfigs(config_space=None):
    if not config_space:
        config_space = global_config_space
    doe = build.full_fact(config_space)
    doe['i'] = doe.index + 1
    doe['NodeTypeStr'] = list(map(lambda nt: config_space['NodeType'][int(nt)], doe['NodeType']))
    return doe

def generateFunctionConfigs(funcName, config_space=None, stackPath='../openFaas/stack.yml',**kwargs):    
    stack = readStack(stackPath)
    functions = stack['functions']
    funcConfig = functions[funcName]
    del functions[funcName]
    
    if not config_space:
        config_space = global_config_space
        
    doe = genConfigs(config_space)
    funcNames = []
    for _, row in doe.iterrows():
        name = f"{funcName}_CPU{row.CPU}_Mem_{row.Mem}_{row.NodeTypeStr}"
        config = funcConfig.copy()
        config['limits'] = {
            'cpu': row.CPU,
            "memory": f"{row.Mem}m"
        }
        config['requests'] = config['limits'].copy()
        functions[name] = config
        funcNames.append(name)
    doe['funcName'] = funcNames
    
    stack['functions'] = functions
    return doe, stack

def runFunction(funcName, data="100", stackYml=None, port=4500, MAXBUF=120, verbose=False):
    if not stackYml: 
        # Assume stack.yml file is same name as function
        stackYml = f"../openFaas/{funcName}.yml"

    logFile = f"/tmp/openFaas/{funcName}.log"
    os.makedirs(os.path.dirname(logFile), exist_ok=True)

    cmd = f"cd {os.path.dirname(stackYml)}; faas-cli local-run {funcName} -f {stackYml} -p {port} --quiet 2>&1 | tee {logFile}"
    if verbose:
        print("running command: ")
        print(cmd)
    process = subprocess.Popen(cmd, bufsize=MAXBUF, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    duration = None
    for line in process.stdout:
        if line:
            if verbose:
                print(line)
            if re.search("Listening on port: 8080", line):
                # TODO use requests and send payload from input 
                command = f"curl http://0.0.0.0:{port} -X POST  --data '{data}' >/dev/null 2>&1"
                if verbose:
                    print(f"Running command: {command}")
                os.system(command)
            elif (match:= re.search("Duration: ((?:\d|.)+)s", line)):
                duration = float(match.groups()[0])
                break

    if not process.poll():
        process.kill()
    # Ensure all subprocesses are killed for this function
    os.system("""kill -9 $(ps -ef | grep -e "faas-cli local-run %s" -e "docker run --name %s" -e "run.*%s" | awk '{print $2}')""" % (funcName,funcName,funcName))
    print(f"duration of {funcName}: {duration}s")
    return duration

def getTime(funcName, **kwargs):
    return runFunction(funcName, **kwargs)

def getTimes(doe, stackYml="../openFaas/matmul_gen.yml", data="1000", verbose=False, **kwargs):
    if verbose:
        print(f"Getting times use data: {data}")

    def wrapGetTime(row):
        return getTime(row.funcName, port=(4500+row.i), data=data, stackYml=stackYml, verbose=verbose)

    results = executeThreaded(wrapGetTime, doe.itertuples(index=False), max_workers=8)
    doe['time'] = results
    return doe

def parseArgs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stack", type=str,
                        help="The .yml file with the initial OpenFaaS function definition")
    parser.add_argument("-f", "--func", type=str,
                        help="The function within the .yml file to optimize. Defaults to first function in stack.yml")
    parser.add_argument("-o", "--outStack", type=str,
                        help="The output .yml path where the configurations will be stored. Defaults to <stack>_gen.yml")
    parser.add_argument("-c", "--cpu", type=float, nargs="+",
                        help=f"The range of cpu values to test for. Defaults to {global_config_space['CPU']}")
    parser.add_argument("-m", "--mem", type=float, nargs="+",
                        help=f"The range of memory values to test for.Defaults to {global_config_space['Mem']}")
    parser.add_argument("-nt", "--nodeType", type=str, nargs="+",
                        help=f"The list of NodeTypes to test for. Defaults to {global_config_space['NodeType']}")
    parser.add_argument("-d", "--data", type=str,
                        help=f"The input data to pass to the function to get the estimated runtime")
    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction,
                        help=f"Verbose output, helpful for debugging. Defaults to False")
    args = parser.parse_args()

    config_space = {
        'CPU': args.cpu or global_config_space.get("CPU"),
        'Mem': args.mem or global_config_space.get("Mem"),
        'NodeType': args.nodeType or global_config_space.get("NodeType")
    }

    if not args.outStack:
        head, tail = os.path.splitext(args.stack)
        args.outStack = f"{head}_gen{tail}"
    elif os.path.isdir(args.outStack):
        head, tail = os.path.splitext(os.path.basename(args.stack))
        args.outStack = os.path.join(args.outStack, f"{head}_gen{tail}")

    stack = readStack(args.stack)
    if not stack.get('functions', False):
        return f"{args.stack} not valid"

    if not args.func:
        # Assume at least 1 function available in this stack
        args.func = next(iter(stack['functions'].items()))[0]

    finalArgs =  {
        "stackPath": args.stack,
        "genStackPath": args.outStack,
        "funcName": args.func,
        "config_space": config_space,
        "data": args.data or "1000",
        "verbose": args.verbose or False
    }
    if finalArgs['verbose']:
        print(f"finalArgs: {finalArgs}")
    return finalArgs

def main():
    # Get arguments 
    args = parseArgs()

    # Generate a new stack file with new configs
    doe, stack = generateFunctionConfigs(**args)
    writeStack(stack, args['genStackPath'])
    print(f"Generated {len(doe)} configurations, testing now")

    doe = getTimes(doe, stackYml=args['genStackPath'], **args)
    doe.sort_values(by=["time"], inplace=True)
    print(tabulate(doe[['CPU','Mem','NodeTypeStr', 'time']].reset_index(drop=True), headers='keys', tablefmt='psql'))

    best = doe.iloc[0]
    print(f"Top Recommendation config: CPU :{best.CPU}, Mem: {best.Mem}, NodeType: {best.NodeTypeStr} which had a final time of: {best.time}s")
    return args, doe, stack

if __name__ == "__main__":
    main()
    # Clears up any docker instances
    os.system("docker stop $(docker ps | grep fwatchdog | awk '{print $1}') > /dev/null 2>&1")

