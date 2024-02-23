"""Allow to parse the variables and templates using drain parser. Main function is get_parsing_drainparser. Note that if you want the template you can make again a function similar to get_parsing_drainparser by using parse and get_templates_variables_per_lines"""

import re
from typing import *# type: ignore
import uuid
from collections import OrderedDict

TokenLength = int
Token = str
Wildcard = str
NodeMapping = Dict[Union[TokenLength, Token, Wildcard], "Node"]


class TemplateGroup(TypedDict):
    template: List[str]
    lines: List[Tuple[int, str]]
    id: str

class ParsedLine(TypedDict):
    """Represents a parsed line
    
    - template: str, the template of this line
    - variables: List[str], the variables associated with this line
    """
    template: str
    variables: List[str]


default_depth: int = 2
default_max_child: int = 3
default_similarity_threshold: float = 0.4


class Logcluster:
    """Class to represent log clusters inside tree leaves"""

    def __init__(self, logTemplate: Optional[List[str]] = None, logIDL=None):
        if logTemplate is None:
            logTemplate = []
        self.logTemplate = logTemplate
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL
        self.id = str(uuid.uuid4())
    def __str__(self) -> str:
        return f"{self.logTemplate=}\n{self.logIDL=}\n{self.id=}"


class Node:
    """
    Class to represent the node of the tree
    # Arguments
    - childD: Optional[Dict[Union[TokenLength, Token, Wildcard],"Node"]], dictionary where the key represents lengths, token or wildcards, Node is the child found in the token
    - depth: int, depth of the current node
    - digitOrtoken: Union[str, int, None], the concrete data represented by the node
    """

    def __init__(
        self,
        childD: Optional[Union[NodeMapping, List[Logcluster]]] = None,
        depth=0,
        digitOrtoken: Optional[Union[str, int]] = None,
    ):
        if childD is None:
            childD = dict()
        self.childD: Union[NodeMapping, List[Logcluster]] = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken


def hasNumbers(s: str) -> bool:
    return any(char.isdigit() for char in s)


def treeSearch(
    rn: Node,
    seq: List[str],
    depth: int = default_depth,
    similarity_threshold: float = default_similarity_threshold,
) -> Optional[Logcluster]:
    """Method to search for a sequence <<seq>> inside a node <<rn>>

    # Arguments:
        - rn: Node, root node to start the search from
        - seq: List[str], the sequence of tokens to process
    """
    retLogClust = None

    # Quick check by length of the tokens if it is in the root node
    seqLen: TokenLength = len(seq)
    # If not there are no match, so we return None
    if seqLen not in rn.childD:
        return retLogClust

    parentn: Node = rn.childD[seqLen]  # type: ignore

    currentDepth = 1
    for token in seq:
        if currentDepth >= depth or currentDepth > seqLen:
            break

        if token in parentn.childD:
            parentn = parentn.childD[token]  # type: ignore
        elif "<*>" in parentn.childD:
            parentn = parentn.childD["<*>"]  # type: ignore
        else:
            return retLogClust
        currentDepth += 1

    logClustL = parentn.childD

    retLogClust = fastMatch(logClustL, seq, similarity_threshold=similarity_threshold)

    return retLogClust


def addSeqToPrefixTree(
    rn: Node,
    logClust: Logcluster,
    depth: int = default_depth,
    max_child: int = default_max_child,
):
    """Method to add a new sequence as a log cluster to the prefix tree, as it was not found before with treeSearch

    # Arguments:
    - rn: Node, the root node of the file
    - logClust: Logcluster, the new log cluster to add

    """
    seqLen = len(logClust.logTemplate)
    ## If we dont have a node representing sequences of the same number of tokens. Then we get this node
    if seqLen not in rn.childD:
        assert isinstance(rn.childD, dict)
        firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
        rn.childD[seqLen] = firtLayerNode
    else:
        firtLayerNode = rn.childD[seqLen]

    parentn = firtLayerNode
    assert isinstance(parentn, Node)
    currentDepth = 1
    for token in logClust.logTemplate:
        ## Add current log cluster to the leaf node
        if currentDepth >= depth or currentDepth > seqLen:
            # assert isinstance(parentn.childD, list)
            if len(parentn.childD) == 0:
                parentn.childD = [logClust]
            else:
                parentn.childD.append(logClust)
            break

        ## If token not matched in this layer of existing tree.
        if token not in parentn.childD:
            assert isinstance(parentn.childD, dict)
            if not hasNumbers(token):
                if "<*>" in parentn.childD:
                    if len(parentn.childD) < max_child:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                        parentn.childD[token] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD["<*>"]
                else:
                    if len(parentn.childD) + 1 < max_child:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                        parentn.childD[token] = newNode
                        parentn = newNode
                    elif len(parentn.childD) + 1 == max_child:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                        parentn.childD["<*>"] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD["<*>"]

            else:
                if "<*>" not in parentn.childD:
                    newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                    parentn.childD["<*>"] = newNode
                    parentn = newNode
                else:
                    parentn = parentn.childD["<*>"]

        ## If the token is matched
        else:
            assert isinstance(parentn.childD, dict)
            parentn = parentn.childD[token]

        ## Goes further in depth
        currentDepth += 1


def seqDist(seq1: List[str], seq2: List[str]) -> Tuple[float, int]:
    """Method to measure the SimSeq between two sequences seq1 and seq2

    # Return
    - Tuple[float, int]
        - retVal: float, the SimSeq between the two sequences
        - numOfPar: the number of tokens <*>  in seq1
    """
    assert len(seq1) == len(seq2)
    simTokens = 0
    numOfPar = 0

    for token1, token2 in zip(seq1, seq2):
        if token1 == "<*>":
            numOfPar += 1
            continue
        if token1 == token2:
            simTokens += 1

    retVal = float(simTokens) / len(seq1)

    return retVal, numOfPar


def fastMatch(
    logClustL: List[Logcluster],
    seq: List[str],
    similarity_threshold: float = default_similarity_threshold,
) -> Optional[Logcluster]:
    """Method to check the maximum similarity threshold between leaves/log clusters.
    Chooses the closest cluster (based on seqDist) for which the template is the closest to seq and above the threshold st

    # Arguments
    - logClustL: List[Logcluster], the clusters to compare the sequence to
    - seq: List[str], the sequence to analyse

    # Returns
    - Optional[Logcluster], either the maximum similarity cluster if above the threshold or None if no cluster found/above the threshold
    """
    retLogClust = None

    maxSim = -1
    maxNumOfPara = -1
    maxClust = None

    for logClust in logClustL:
        curSim, curNumOfPara = seqDist(logClust.logTemplate, seq)
        if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
            maxSim = curSim
            maxNumOfPara = curNumOfPara
            maxClust = logClust

    if maxSim >= similarity_threshold:
        retLogClust = maxClust

    return retLogClust


def getTemplate(seq1: List[str], seq2: List[str]) -> List[str]:
    """Method to get the template of <<seq1>> based on its similar words with seq2

    Example:
    seq1: I love books
    seq2: I love food
    Returns: I love <*>

    # Arguments
    - seq1: List[str], the first sequence to compare
    - seq2: List[str], the second sequence to compare

    # Return
    - List[str], the template, common tokens between the sequences and wildcards <*> for any different tokens
    """
    assert len(seq1) == len(seq2)
    retVal = []

    i = 0
    for word in seq1:
        if word == seq2[i]:
            retVal.append(word)
        else:
            retVal.append("<*>")

        i += 1

    return retVal


def get_parameter_list(template: str, text: str) -> List[str]:
    """Get the list of parameters from the template and the full text"""
    template_regex = re.sub(r"<.{1,5}>", "<*>", template)
    if "<*>" not in template_regex:
        return []
    template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)

    template_regex = re.sub(r"\\\s+", "\\\s+", template_regex)  # type: ignore

    template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"  # type: ignore
    parameter_list = re.findall(template_regex, text)
    parameter_list = parameter_list[0] if parameter_list else ()
    parameter_list = (
        list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
    )
    return parameter_list


def parse(
    preprocessed_texts: List[str],
    reg_expressions: List[str],
    depth: int = default_depth,
    max_child: int = default_max_child,
    similarity_threshold: float = default_similarity_threshold,
) -> Tuple[List[Logcluster], Dict[str, TemplateGroup]]:
    """Parse the logs and update the templates iteratively
    
    # Arguments
    - preprocessed_texts: List[str], list of texts preprocessed already
    - depth: int = default_depth, 
    - max_child: int = default_max_child,
    - similarity_threshold: float = default_similarity_threshold,
    """
    rootNode = Node()
    logCluL = []
    # prepare the mapping of each template to each line
    templates: Dict[str, TemplateGroup] = {}
    for i, line in enumerate(preprocessed_texts):
        logID = i

        ## Tokenization by splits
        logmessage = preprocess(line,reg_expressions)
        logmessageL = logmessage.strip().split()

        matchCluster = treeSearch(
            rootNode,
            logmessageL,
            depth=depth,
            similarity_threshold=similarity_threshold,
        )

        ## Match no existing log cluster
        if matchCluster is None:
            matchCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
            logCluL.append(matchCluster)
            addSeqToPrefixTree(rootNode, matchCluster, depth=depth, max_child=max_child)
        ## Adds the new log message to the existing cluster
        else:
            newTemplate = getTemplate(logmessageL, matchCluster.logTemplate)
            matchCluster.logIDL.append(logID)
            if " ".join(newTemplate) != " ".join(matchCluster.logTemplate):
                matchCluster.logTemplate = newTemplate
        # With defaultdict we can directly append the line id to the template list
        if matchCluster.id not in templates:
            templates[matchCluster.id] = TemplateGroup(
                id=matchCluster.id, lines=[], template=matchCluster.logTemplate
            )
        templates[matchCluster.id]['lines'].append((i,line))
        templates[matchCluster.id]['template'] = matchCluster.logTemplate
    return logCluL, templates


def get_templates_variables_per_lines(
    templates: Dict[str, TemplateGroup]
) -> List[ParsedLine]:
    """Extract the template and variables associated with each line
    
    # Arguments:
    - templates: Dict[str, TemplateGroup], the generated template object that contains the templates groups
    
    # Return
    - List[ParsedLine], for each line the template and the variables
    """
    L = []
    for _, template_group in templates.items():
        template_str = " ".join(template_group["template"])
        for i, l in template_group["lines"]:
            L.append({"line_number": i, "template":template_str, "variables":get_parameter_list(template_str, l), "line":l})
    L.sort(key=lambda x:x['line_number'])
    return [{"template": e["template"], "variables": e["variables"]} for e in L]

def preprocess( line: str, reg_expressions):
    """Method to preprocess file using regex: replace in the line all self.rex regex specified by <*>"""
    for currentRex in reg_expressions:
        line = re.sub(currentRex, '<*>', line)
    return line

def get_parsing_drainparser(
    events: List[str],
    reg_expressions: List[str],
    depth: int = default_depth,
    similarity_threshold: float = default_similarity_threshold,
    max_children: int = default_max_child,
) -> List[ParsedLine]:
    """From the list of log lines, returns for each line the emplate and the variables
    
    # Arguments:
    - events: List[str], the list of log lines
    - depth: int = default_depth, the depth of the log tree parser
    - similarity_threshold: float = default_similarity_threshold, the minimum similarity between two lines to create a template, must be between 0 and 1
    - max_children: int = default_max_child, the maximum number of children for one template
    
    # Return
    - List[ParsedLine], for each line the template and the variables
    """
    # load the data
    L, templates = parse(
        events,
        depth=depth,
        max_child=max_children,
        similarity_threshold=similarity_threshold,
        reg_expressions=reg_expressions,
    )
    variables = get_templates_variables_per_lines(templates)
    return variables
